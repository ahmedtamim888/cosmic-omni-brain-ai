"""
🎓 Educational Telegram Bot
Interactive learning platform with quizzes, lessons, progress tracking, and AI-powered assistance
"""

import asyncio
import logging
import json
import random
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ParseMode
import sqlite3
from dataclasses import dataclass, asdict
from config import Config

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class User:
    user_id: int
    username: str
    first_name: str
    last_name: str
    language_code: str
    level: str = "beginner"
    points: int = 0
    streak: int = 0
    last_activity: str = ""
    created_at: str = ""

@dataclass
class QuizQuestion:
    id: int
    category: str
    difficulty: str
    question: str
    options: List[str]
    correct_answer: int
    explanation: str
    points: int

@dataclass
class Lesson:
    id: int
    title: str
    category: str
    difficulty: str
    content: str
    duration: int  # in minutes
    prerequisites: List[str]

class DatabaseManager:
    """Manages SQLite database for user data and educational content"""
    
    def __init__(self, db_path: str = "education_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    language_code TEXT,
                    level TEXT DEFAULT 'beginner',
                    points INTEGER DEFAULT 0,
                    streak INTEGER DEFAULT 0,
                    last_activity TEXT,
                    created_at TEXT
                )
            """)
            
            # Quiz results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    question_id INTEGER,
                    answer INTEGER,
                    is_correct BOOLEAN,
                    points_earned INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # User progress table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    lesson_id INTEGER,
                    completed BOOLEAN DEFAULT FALSE,
                    completion_date TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Study sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    category TEXT,
                    duration INTEGER,
                    questions_answered INTEGER,
                    correct_answers INTEGER,
                    timestamp TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            conn.commit()
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return User(*row)
            return None
    
    def create_user(self, user: User) -> bool:
        """Create or update user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO users 
                    (user_id, username, first_name, last_name, language_code, level, points, streak, last_activity, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user.user_id, user.username, user.first_name, user.last_name,
                    user.language_code, user.level, user.points, user.streak,
                    user.last_activity, user.created_at
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                return False
    
    def update_user_points(self, user_id: int, points: int) -> bool:
        """Update user points"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    UPDATE users SET points = points + ?, last_activity = ?
                    WHERE user_id = ?
                """, (points, datetime.now().isoformat(), user_id))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Error updating points: {e}")
                return False
    
    def save_quiz_result(self, user_id: int, question_id: int, answer: int, is_correct: bool, points: int):
        """Save quiz result"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO quiz_results (user_id, question_id, answer, is_correct, points_earned, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, question_id, answer, is_correct, points, datetime.now().isoformat()))
            conn.commit()

class EducationalBot:
    """Main Educational Telegram Bot Class"""
    
    def __init__(self):
        self.bot_token = Config.TELEGRAM_BOT_TOKEN
        self.app = None
        self.db = DatabaseManager()
        self.active_quizzes = {}  # Store active quiz sessions
        self.quiz_questions = self._load_quiz_questions()
        self.lessons = self._load_lessons()
        
    def _load_quiz_questions(self) -> List[QuizQuestion]:
        """Load quiz questions from data"""
        return [
            QuizQuestion(
                id=1,
                category="Mathematics",
                difficulty="beginner",
                question="What is 2 + 2?",
                options=["3", "4", "5", "6"],
                correct_answer=1,
                explanation="2 + 2 = 4. This is basic addition.",
                points=10
            ),
            QuizQuestion(
                id=2,
                category="Mathematics",
                difficulty="beginner",
                question="What is 5 × 3?",
                options=["13", "15", "18", "20"],
                correct_answer=1,
                explanation="5 × 3 = 15. Multiplication means repeated addition: 5 + 5 + 5 = 15.",
                points=10
            ),
            QuizQuestion(
                id=3,
                category="Science",
                difficulty="beginner",
                question="What is the largest planet in our solar system?",
                options=["Earth", "Jupiter", "Saturn", "Mars"],
                correct_answer=1,
                explanation="Jupiter is the largest planet in our solar system with a diameter of about 142,984 km.",
                points=15
            ),
            QuizQuestion(
                id=4,
                category="History",
                difficulty="intermediate",
                question="In which year did World War II end?",
                options=["1944", "1945", "1946", "1947"],
                correct_answer=1,
                explanation="World War II ended in 1945 with the surrender of Japan in September.",
                points=20
            ),
            QuizQuestion(
                id=5,
                category="Geography",
                difficulty="beginner",
                question="What is the capital of France?",
                options=["London", "Paris", "Berlin", "Madrid"],
                correct_answer=1,
                explanation="Paris is the capital and largest city of France.",
                points=10
            ),
            QuizQuestion(
                id=6,
                category="Science",
                difficulty="intermediate",
                question="What is the chemical symbol for water?",
                options=["H2O", "CO2", "NaCl", "O2"],
                correct_answer=0,
                explanation="H2O represents water - two hydrogen atoms bonded to one oxygen atom.",
                points=15
            ),
            QuizQuestion(
                id=7,
                category="Mathematics",
                difficulty="advanced",
                question="What is the derivative of x²?",
                options=["x", "2x", "x²", "2x²"],
                correct_answer=1,
                explanation="The derivative of x² is 2x using the power rule: d/dx(xⁿ) = nxⁿ⁻¹.",
                points=25
            ),
            QuizQuestion(
                id=8,
                category="Literature",
                difficulty="intermediate",
                question="Who wrote 'Romeo and Juliet'?",
                options=["Charles Dickens", "William Shakespeare", "Mark Twain", "Jane Austen"],
                correct_answer=1,
                explanation="William Shakespeare wrote the famous tragedy 'Romeo and Juliet' around 1594-1596.",
                points=15
            )
        ]
    
    def _load_lessons(self) -> List[Lesson]:
        """Load educational lessons"""
        return [
            Lesson(
                id=1,
                title="Introduction to Basic Math",
                category="Mathematics",
                difficulty="beginner",
                content="""
🔢 <b>Basic Mathematics - Addition & Subtraction</b>

<b>What is Addition?</b>
Addition (+) means combining numbers to get a larger number.
Example: 3 + 2 = 5

<b>What is Subtraction?</b>
Subtraction (-) means taking away from a number.
Example: 5 - 2 = 3

<b>Practice Tips:</b>
• Start with small numbers
• Use your fingers to count
• Practice daily for 10 minutes
• Try word problems

<b>Key Rules:</b>
• a + b = b + a (commutative)
• a + 0 = a (adding zero)
• a - a = 0 (subtracting same number)
                """,
                duration=15,
                prerequisites=[]
            ),
            Lesson(
                id=2,
                title="Solar System Basics",
                category="Science",
                difficulty="beginner",
                content="""
🌌 <b>Our Solar System</b>

<b>What is the Solar System?</b>
The Solar System consists of the Sun and all objects that orbit around it.

<b>The Planets in Order:</b>
1. Mercury ☿️ - Closest to Sun
2. Venus ♀️ - Hottest planet
3. Earth 🌍 - Our home
4. Mars ♂️ - The red planet
5. Jupiter ♃ - Largest planet
6. Saturn ♄ - Has beautiful rings
7. Uranus ♅ - Tilted sideways
8. Neptune ♆ - Windiest planet

<b>Fun Facts:</b>
• The Sun is a star ⭐
• Earth is the only planet with known life
• Jupiter is bigger than all other planets combined
• Saturn's rings are made of ice and rock
                """,
                duration=20,
                prerequisites=[]
            ),
            Lesson(
                id=3,
                title="World Geography Basics",
                category="Geography",
                difficulty="beginner",
                content="""
🌍 <b>World Geography - Continents & Oceans</b>

<b>The 7 Continents:</b>
1. Asia 🏔️ - Largest continent
2. Africa 🦁 - Has the Sahara Desert
3. North America 🍁 - Contains USA, Canada, Mexico
4. South America 🌴 - Contains Amazon rainforest
5. Antarctica 🐧 - Coldest continent
6. Europe 🏰 - Many small countries
7. Australia/Oceania 🦘 - Smallest continent

<b>The 5 Oceans:</b>
1. Pacific 🌊 - Largest ocean
2. Atlantic 🚢 - Between Americas and Europe/Africa
3. Indian 🏝️ - Around India
4. Arctic ❄️ - Coldest ocean
5. Southern 🐋 - Around Antarctica

<b>Study Tips:</b>
• Use maps and globes
• Learn one continent per week
• Practice with online quizzes
                """,
                duration=25,
                prerequisites=[]
            )
        ]
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        chat_id = update.effective_chat.id
        
        # Create or update user in database
        user_obj = User(
            user_id=user.id,
            username=user.username or "",
            first_name=user.first_name or "",
            last_name=user.last_name or "",
            language_code=user.language_code or "en",
            created_at=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat()
        )
        
        existing_user = self.db.get_user(user.id)
        if not existing_user:
            self.db.create_user(user_obj)
            welcome_type = "new"
        else:
            welcome_type = "returning"
            user_obj = existing_user
        
        if welcome_type == "new":
            welcome_message = f"""
🎓 <b>Welcome to Educational Bot!</b> 📚

Hello <b>{user.first_name}</b>! I'm your personal learning assistant.

🌟 <b>What I can help you with:</b>
• 📝 Interactive quizzes in multiple subjects
• 📖 Structured lessons and tutorials  
• 📊 Track your learning progress
• 🏆 Earn points and maintain streaks
• 🎯 Personalized difficulty levels
• 📈 Detailed performance analytics

🎯 <b>Available Subjects:</b>
• Mathematics 🔢
• Science 🔬
• History 📜
• Geography 🌍
• Literature 📚

Ready to start your learning journey?
            """
        else:
            welcome_message = f"""
🎓 <b>Welcome back, {user.first_name}!</b> 📚

📊 <b>Your Progress:</b>
• Level: {user_obj.level.title()}
• Points: {user_obj.points} 🏆
• Current Streak: {user_obj.streak} days 🔥

Continue your learning adventure!
            """
        
        keyboard = [
            [InlineKeyboardButton("📝 Start Quiz", callback_data="start_quiz")],
            [InlineKeyboardButton("📖 Browse Lessons", callback_data="browse_lessons")],
            [InlineKeyboardButton("📊 My Progress", callback_data="my_progress"),
             InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = query.from_user.id
        
        if data == "start_quiz":
            await self.show_quiz_categories(query)
        elif data == "browse_lessons":
            await self.show_lesson_categories(query)
        elif data == "my_progress":
            await self.show_user_progress(query)
        elif data == "leaderboard":
            await self.show_leaderboard(query)
        elif data == "settings":
            await self.show_settings(query)
        elif data == "help":
            await self.show_help(query)
        elif data.startswith("quiz_category_"):
            category = data.replace("quiz_category_", "")
            await self.start_quiz_session(query, category)
        elif data.startswith("lesson_category_"):
            category = data.replace("lesson_category_", "")
            await self.show_lessons_in_category(query, category)
        elif data.startswith("lesson_"):
            lesson_id = int(data.replace("lesson_", ""))
            await self.show_lesson(query, lesson_id)
        elif data.startswith("answer_"):
            await self.handle_quiz_answer(query, data)
        elif data == "next_question":
            await self.next_quiz_question(query)
        elif data == "end_quiz":
            await self.end_quiz_session(query)
        elif data == "back_to_main":
            await self.show_main_menu(query)
    
    async def show_quiz_categories(self, query):
        """Show available quiz categories"""
        categories = list(set(q.category for q in self.quiz_questions))
        
        message = """
📝 <b>Choose a Quiz Category</b>

Select a subject to test your knowledge:
        """
        
        keyboard = []
        for category in categories:
            count = len([q for q in self.quiz_questions if q.category == category])
            keyboard.append([InlineKeyboardButton(
                f"{category} ({count} questions)", 
                callback_data=f"quiz_category_{category}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def start_quiz_session(self, query, category: str):
        """Start a quiz session for a specific category"""
        user_id = query.from_user.id
        
        # Get questions for this category
        category_questions = [q for q in self.quiz_questions if q.category == category]
        random.shuffle(category_questions)
        
        # Initialize quiz session
        self.active_quizzes[user_id] = {
            'category': category,
            'questions': category_questions,
            'current_question': 0,
            'score': 0,
            'total_questions': len(category_questions),
            'start_time': datetime.now(),
            'answers': []
        }
        
        await self.show_current_question(query)
    
    async def show_current_question(self, query):
        """Show the current quiz question"""
        user_id = query.from_user.id
        
        if user_id not in self.active_quizzes:
            await query.edit_message_text("❌ No active quiz session found.")
            return
        
        quiz_session = self.active_quizzes[user_id]
        current_q_index = quiz_session['current_question']
        
        if current_q_index >= len(quiz_session['questions']):
            await self.end_quiz_session(query)
            return
        
        question = quiz_session['questions'][current_q_index]
        
        progress = current_q_index + 1
        total = quiz_session['total_questions']
        
        message = f"""
📝 <b>Quiz: {quiz_session['category']}</b>
Progress: {progress}/{total} | Score: {quiz_session['score']}

<b>Question {progress}:</b>
{question.question}

<b>Difficulty:</b> {question.difficulty.title()} ({question.points} points)
        """
        
        keyboard = []
        for i, option in enumerate(question.options):
            keyboard.append([InlineKeyboardButton(
                f"{chr(65 + i)}. {option}",
                callback_data=f"answer_{question.id}_{i}"
            )])
        
        keyboard.append([InlineKeyboardButton("❌ End Quiz", callback_data="end_quiz")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def handle_quiz_answer(self, query, answer_data: str):
        """Handle quiz answer selection"""
        user_id = query.from_user.id
        
        if user_id not in self.active_quizzes:
            await query.edit_message_text("❌ No active quiz session found.")
            return
        
        # Parse answer data
        parts = answer_data.split("_")
        question_id = int(parts[1])
        selected_answer = int(parts[2])
        
        quiz_session = self.active_quizzes[user_id]
        current_question = quiz_session['questions'][quiz_session['current_question']]
        
        # Check if answer is correct
        is_correct = selected_answer == current_question.correct_answer
        points_earned = current_question.points if is_correct else 0
        
        # Update quiz session
        quiz_session['answers'].append({
            'question_id': question_id,
            'selected': selected_answer,
            'correct': is_correct,
            'points': points_earned
        })
        
        if is_correct:
            quiz_session['score'] += points_earned
        
        # Save to database
        self.db.save_quiz_result(user_id, question_id, selected_answer, is_correct, points_earned)
        if is_correct:
            self.db.update_user_points(user_id, points_earned)
        
        # Show answer explanation
        result_emoji = "✅" if is_correct else "❌"
        result_text = "Correct!" if is_correct else "Incorrect"
        
        message = f"""
{result_emoji} <b>{result_text}</b>

<b>Your answer:</b> {chr(65 + selected_answer)}. {current_question.options[selected_answer]}
<b>Correct answer:</b> {chr(65 + current_question.correct_answer)}. {current_question.options[current_question.correct_answer]}

<b>Explanation:</b>
{current_question.explanation}

<b>Points earned:</b> {points_earned}/{current_question.points}
        """
        
        keyboard = [
            [InlineKeyboardButton("➡️ Next Question", callback_data="next_question")],
            [InlineKeyboardButton("❌ End Quiz", callback_data="end_quiz")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def next_quiz_question(self, query):
        """Move to next quiz question"""
        user_id = query.from_user.id
        
        if user_id not in self.active_quizzes:
            await query.edit_message_text("❌ No active quiz session found.")
            return
        
        quiz_session = self.active_quizzes[user_id]
        quiz_session['current_question'] += 1
        
        await self.show_current_question(query)
    
    async def end_quiz_session(self, query):
        """End current quiz session and show results"""
        user_id = query.from_user.id
        
        if user_id not in self.active_quizzes:
            await query.edit_message_text("❌ No active quiz session found.")
            return
        
        quiz_session = self.active_quizzes[user_id]
        
        # Calculate final statistics
        total_questions = len(quiz_session['answers'])
        correct_answers = sum(1 for a in quiz_session['answers'] if a['correct'])
        total_points = quiz_session['score']
        accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        # Determine performance level
        if accuracy >= 90:
            performance = "🌟 Excellent!"
        elif accuracy >= 80:
            performance = "🎉 Great job!"
        elif accuracy >= 70:
            performance = "👍 Good work!"
        elif accuracy >= 60:
            performance = "📈 Keep practicing!"
        else:
            performance = "💪 Don't give up!"
        
        duration = datetime.now() - quiz_session['start_time']
        duration_minutes = duration.total_seconds() / 60
        
        message = f"""
🏁 <b>Quiz Complete!</b>

<b>Category:</b> {quiz_session['category']}
<b>Performance:</b> {performance}

📊 <b>Results:</b>
• Questions answered: {total_questions}
• Correct answers: {correct_answers}
• Accuracy: {accuracy:.1f}%
• Points earned: {total_points}
• Time taken: {duration_minutes:.1f} minutes

🎯 <b>Recommendations:</b>
        """
        
        if accuracy < 70:
            message += "\n• Review the lesson materials"
            message += "\n• Practice more questions in this category"
        elif accuracy < 90:
            message += "\n• Try more advanced questions"
            message += "\n• Explore related topics"
        else:
            message += "\n• Try a different category"
            message += "\n• Challenge yourself with advanced level"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Retake Quiz", callback_data=f"quiz_category_{quiz_session['category']}")],
            [InlineKeyboardButton("📝 Try Another Category", callback_data="start_quiz")],
            [InlineKeyboardButton("📖 Study Lessons", callback_data="browse_lessons")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Clean up active quiz
        del self.active_quizzes[user_id]
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def show_lesson_categories(self, query):
        """Show available lesson categories"""
        categories = list(set(lesson.category for lesson in self.lessons))
        
        message = """
📖 <b>Browse Lessons</b>

Choose a subject to start learning:
        """
        
        keyboard = []
        for category in categories:
            lesson_count = len([l for l in self.lessons if l.category == category])
            keyboard.append([InlineKeyboardButton(
                f"{category} ({lesson_count} lessons)",
                callback_data=f"lesson_category_{category}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def show_lessons_in_category(self, query, category: str):
        """Show lessons in a specific category"""
        category_lessons = [l for l in self.lessons if l.category == category]
        
        message = f"""
📖 <b>{category} Lessons</b>

Available lessons in this category:
        """
        
        keyboard = []
        for lesson in category_lessons:
            difficulty_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(lesson.difficulty, "⚪")
            keyboard.append([InlineKeyboardButton(
                f"{difficulty_emoji} {lesson.title} ({lesson.duration}min)",
                callback_data=f"lesson_{lesson.id}"
            )])
        
        keyboard.append([InlineKeyboardButton("🔙 Back to Categories", callback_data="browse_lessons")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def show_lesson(self, query, lesson_id: int):
        """Show a specific lesson"""
        lesson = next((l for l in self.lessons if l.id == lesson_id), None)
        
        if not lesson:
            await query.edit_message_text("❌ Lesson not found.")
            return
        
        difficulty_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}.get(lesson.difficulty, "⚪")
        
        message = f"""
📖 <b>{lesson.title}</b>
{difficulty_emoji} {lesson.difficulty.title()} | ⏱️ {lesson.duration} minutes

{lesson.content}

🎯 <b>What's Next?</b>
Take a quiz to test your understanding!
        """
        
        keyboard = [
            [InlineKeyboardButton("📝 Take Quiz", callback_data=f"quiz_category_{lesson.category}")],
            [InlineKeyboardButton("✅ Mark as Complete", callback_data=f"complete_lesson_{lesson_id}")],
            [InlineKeyboardButton("🔙 Back to Lessons", callback_data=f"lesson_category_{lesson.category}")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def show_user_progress(self, query):
        """Show user's learning progress"""
        user_id = query.from_user.id
        user = self.db.get_user(user_id)
        
        if not user:
            await query.edit_message_text("❌ User not found.")
            return
        
        # Calculate some statistics (you'd get these from database in real implementation)
        total_quizzes = 5  # Placeholder
        total_lessons = 3  # Placeholder
        
        message = f"""
📊 <b>Your Learning Progress</b>

👤 <b>Profile:</b>
• Level: {user.level.title()}
• Total Points: {user.points} 🏆
• Current Streak: {user.streak} days 🔥

📈 <b>Statistics:</b>
• Quizzes completed: {total_quizzes}
• Lessons finished: {total_lessons}
• Average accuracy: 85.2% 📊
• Time spent learning: 2.5 hours ⏰

🎯 <b>Goals for this week:</b>
• Complete 3 more quizzes
• Finish Mathematics basics
• Maintain daily streak

Keep up the great work! 🌟
        """
        
        keyboard = [
            [InlineKeyboardButton("📝 Continue Learning", callback_data="start_quiz")],
            [InlineKeyboardButton("🏆 View Achievements", callback_data="achievements")],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def show_help(self, query):
        """Show help information"""
        message = """
❓ <b>How to Use Educational Bot</b>

🎯 <b>Getting Started:</b>
• Use /start to begin your learning journey
• Choose your preferred subjects
• Set your difficulty level

📝 <b>Taking Quizzes:</b>
• Select a category that interests you
• Answer multiple-choice questions
• Get instant feedback and explanations
• Earn points for correct answers

📖 <b>Learning from Lessons:</b>
• Browse lessons by category
• Study at your own pace
• Practice with related quizzes
• Track your completion progress

🏆 <b>Progress & Gamification:</b>
• Earn points for every correct answer
• Maintain daily learning streaks
• Unlock achievements and badges
• Compete on the leaderboard

⚙️ <b>Commands:</b>
/start - Welcome message and main menu
/progress - View your learning progress
/quiz - Start a quick quiz
/help - Show this help message

📧 <b>Support:</b>
If you need help, just send me a message!
        """
        
        keyboard = [
            [InlineKeyboardButton("🎓 Start Learning", callback_data="start_quiz")],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        message_text = update.message.text.lower()
        
        # Simple AI-like responses
        if any(word in message_text for word in ['help', 'how', 'what', 'explain']):
            await update.message.reply_text(
                "🤖 I'm here to help you learn! Use the buttons in the main menu to:\n\n"
                "📝 Take interactive quizzes\n"
                "📖 Read educational lessons\n"
                "📊 Track your progress\n\n"
                "What would you like to do today?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🎓 Main Menu", callback_data="back_to_main")]
                ])
            )
        elif any(word in message_text for word in ['math', 'mathematics', 'calculate']):
            await update.message.reply_text(
                "🔢 Great choice! Mathematics is fundamental for many fields.\n\n"
                "I have lessons covering:\n"
                "• Basic arithmetic\n"
                "• Algebra basics\n"
                "• Geometry fundamentals\n\n"
                "Ready to start?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📝 Math Quiz", callback_data="quiz_category_Mathematics")],
                    [InlineKeyboardButton("📖 Math Lessons", callback_data="lesson_category_Mathematics")]
                ])
            )
        elif any(word in message_text for word in ['science', 'physics', 'chemistry', 'biology']):
            await update.message.reply_text(
                "🔬 Science is amazing! Let's explore the wonders of our universe.\n\n"
                "Available topics:\n"
                "• Solar system and space\n"
                "• Basic chemistry\n"
                "• Life sciences\n\n"
                "What interests you most?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("📝 Science Quiz", callback_data="quiz_category_Science")],
                    [InlineKeyboardButton("📖 Science Lessons", callback_data="lesson_category_Science")]
                ])
            )
        else:
            # General response
            await update.message.reply_text(
                "💬 Thanks for your message! I'm designed to help you learn through interactive quizzes and lessons.\n\n"
                "Use the main menu to explore all available features!",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🎓 Main Menu", callback_data="back_to_main")]
                ])
            )
    
    async def show_leaderboard(self, query):
        """Show top learners leaderboard"""
        # In a real implementation, you'd fetch this from database
        message = """
🏆 <b>Learning Leaderboard</b>
<i>Top learners this week</i>

🥇 Alice Johnson - 1,250 points
🥈 Bob Smith - 980 points  
🥉 Carol Davis - 875 points
4️⃣ David Wilson - 720 points
5️⃣ Emma Brown - 650 points

📊 <b>Your Ranking:</b> #12 (420 points)

Keep learning to climb the leaderboard! 🚀
        """
        
        keyboard = [
            [InlineKeyboardButton("📈 View My Stats", callback_data="my_progress")],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def show_settings(self, query):
        """Show user settings"""
        message = """
⚙️ <b>Settings</b>

<b>Current Preferences:</b>
• Difficulty Level: Intermediate
• Daily Goal: 5 questions
• Notifications: Enabled
• Language: English

<b>Customize Your Experience:</b>
        """
        
        keyboard = [
            [InlineKeyboardButton("🎯 Change Difficulty", callback_data="change_difficulty")],
            [InlineKeyboardButton("🎯 Set Daily Goal", callback_data="set_goal")],
            [InlineKeyboardButton("🔔 Notification Settings", callback_data="notifications")],
            [InlineKeyboardButton("🔙 Back to Main Menu", callback_data="back_to_main")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    async def show_main_menu(self, query):
        """Show the main menu"""
        user = query.from_user
        
        # Get user from database if exists
        existing_user = self.db.get_user(user.id)
        
        if existing_user:
            welcome_message = f"""
🎓 <b>Welcome back, {user.first_name}!</b> 📚

📊 <b>Your Progress:</b>
• Level: {existing_user.level.title()}
• Points: {existing_user.points} 🏆
• Current Streak: {existing_user.streak} days 🔥

Continue your learning adventure!
            """
        else:
            welcome_message = f"""
🎓 <b>Educational Bot - Main Menu</b> 📚

Hello <b>{user.first_name}</b>! Ready to learn something new today?

Choose an option below to get started:
            """
        
        keyboard = [
            [InlineKeyboardButton("📝 Start Quiz", callback_data="start_quiz")],
            [InlineKeyboardButton("📖 Browse Lessons", callback_data="browse_lessons")],
            [InlineKeyboardButton("📊 My Progress", callback_data="my_progress"),
             InlineKeyboardButton("🏆 Leaderboard", callback_data="leaderboard")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
             InlineKeyboardButton("❓ Help", callback_data="help")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            welcome_message,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    
    def setup_application(self):
        """Setup the Telegram application with handlers"""
        self.app = Application.builder().token(self.bot_token).build()
        
        # Add command handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        
        # Add callback query handler for buttons
        self.app.add_handler(CallbackQueryHandler(self.button_handler))
        
        # Add message handler for text messages
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Set bot commands for menu
        commands = [
            BotCommand("start", "Start the educational bot"),
            BotCommand("progress", "View your learning progress"),
            BotCommand("quiz", "Take a quick quiz"),
            BotCommand("help", "Get help and instructions")
        ]
        
        asyncio.create_task(self.app.bot.set_my_commands(commands))
    
    async def run(self):
        """Run the bot"""
        if not self.bot_token:
            logger.error("Bot token not configured")
            return
        
        self.setup_application()
        
        logger.info("Starting Educational Telegram Bot...")
        
        # Start the bot
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down bot...")
        finally:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()

# Main execution
if __name__ == "__main__":
    # Create and run the educational bot
    bot = EducationalBot()
    
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {e}")