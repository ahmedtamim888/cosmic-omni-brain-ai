#!/usr/bin/env python3
"""
Test script for Educational Telegram Bot
Verifies all components work correctly before deployment
"""

import unittest
import sqlite3
import os
import sys
from datetime import datetime

# Import our bot components
try:
    from telegram_education_bot import EducationalBot, DatabaseManager, QuizQuestion, Lesson, User
    from config import Config
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class TestEducationalBot(unittest.TestCase):
    """Test cases for Educational Bot functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_db_path = "test_education_bot.db"
        self.db = DatabaseManager(self.test_db_path)
        self.bot = EducationalBot()
        
    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    def test_database_initialization(self):
        """Test database setup and table creation"""
        print("🧪 Testing database initialization...")
        
        # Check if database file was created
        self.assertTrue(os.path.exists(self.test_db_path))
        
        # Check if tables exist
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['users', 'quiz_results', 'user_progress', 'study_sessions']
            for table in expected_tables:
                self.assertIn(table, tables, f"Table '{table}' not found in database")
        
        print("✅ Database initialization test passed!")
    
    def test_user_operations(self):
        """Test user creation and retrieval"""
        print("🧪 Testing user operations...")
        
        # Create test user
        test_user = User(
            user_id=12345,
            username="testuser",
            first_name="Test",
            last_name="User",
            language_code="en",
            level="beginner",
            points=100,
            streak=5,
            last_activity=datetime.now().isoformat(),
            created_at=datetime.now().isoformat()
        )
        
        # Test user creation
        result = self.db.create_user(test_user)
        self.assertTrue(result, "Failed to create user")
        
        # Test user retrieval
        retrieved_user = self.db.get_user(12345)
        self.assertIsNotNone(retrieved_user, "Failed to retrieve user")
        self.assertEqual(retrieved_user.username, "testuser")
        self.assertEqual(retrieved_user.points, 100)
        
        # Test points update
        update_result = self.db.update_user_points(12345, 50)
        self.assertTrue(update_result, "Failed to update user points")
        
        # Verify points were updated
        updated_user = self.db.get_user(12345)
        self.assertEqual(updated_user.points, 150)
        
        print("✅ User operations test passed!")
    
    def test_quiz_questions_loading(self):
        """Test quiz questions are loaded correctly"""
        print("🧪 Testing quiz questions loading...")
        
        questions = self.bot.quiz_questions
        
        # Check if questions were loaded
        self.assertGreater(len(questions), 0, "No quiz questions loaded")
        
        # Check question structure
        for question in questions:
            self.assertIsInstance(question, QuizQuestion)
            self.assertIsInstance(question.id, int)
            self.assertIsInstance(question.question, str)
            self.assertIsInstance(question.options, list)
            self.assertEqual(len(question.options), 4, "Each question should have 4 options")
            self.assertIn(question.correct_answer, range(4), "Correct answer should be 0-3")
            self.assertGreater(question.points, 0, "Questions should have positive points")
        
        # Check categories
        categories = list(set(q.category for q in questions))
        expected_categories = ["Mathematics", "Science", "History", "Geography", "Literature"]
        
        for category in expected_categories:
            self.assertIn(category, categories, f"Category '{category}' not found")
        
        print(f"✅ Quiz questions test passed! Loaded {len(questions)} questions in {len(categories)} categories")
    
    def test_lessons_loading(self):
        """Test lessons are loaded correctly"""
        print("🧪 Testing lessons loading...")
        
        lessons = self.bot.lessons
        
        # Check if lessons were loaded
        self.assertGreater(len(lessons), 0, "No lessons loaded")
        
        # Check lesson structure
        for lesson in lessons:
            self.assertIsInstance(lesson, Lesson)
            self.assertIsInstance(lesson.id, int)
            self.assertIsInstance(lesson.title, str)
            self.assertIsInstance(lesson.content, str)
            self.assertGreater(lesson.duration, 0, "Lessons should have positive duration")
            self.assertIsInstance(lesson.prerequisites, list)
        
        # Check lesson categories
        lesson_categories = list(set(l.category for l in lessons))
        self.assertGreater(len(lesson_categories), 0, "No lesson categories found")
        
        print(f"✅ Lessons test passed! Loaded {len(lessons)} lessons in {len(lesson_categories)} categories")
    
    def test_quiz_result_saving(self):
        """Test saving quiz results"""
        print("🧪 Testing quiz result saving...")
        
        # Create test user first
        test_user = User(
            user_id=67890,
            username="quizuser",
            first_name="Quiz",
            last_name="Taker",
            language_code="en"
        )
        self.db.create_user(test_user)
        
        # Save a quiz result
        self.db.save_quiz_result(
            user_id=67890,
            question_id=1,
            answer=1,
            is_correct=True,
            points=10
        )
        
        # Verify result was saved
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quiz_results WHERE user_id = ?", (67890,))
            result = cursor.fetchone()
            
            self.assertIsNotNone(result, "Quiz result was not saved")
            self.assertEqual(result[1], 67890)  # user_id
            self.assertEqual(result[2], 1)      # question_id
            self.assertEqual(result[3], 1)      # answer
            self.assertEqual(result[4], True)   # is_correct
            self.assertEqual(result[5], 10)     # points_earned
        
        print("✅ Quiz result saving test passed!")
    
    def test_configuration(self):
        """Test bot configuration"""
        print("🧪 Testing bot configuration...")
        
        # Check if bot token is configured
        self.assertIsNotNone(Config.TELEGRAM_BOT_TOKEN, "Bot token not configured")
        self.assertNotEqual(Config.TELEGRAM_BOT_TOKEN, "", "Bot token is empty")
        
        # Check bot initialization
        self.assertIsNotNone(self.bot.bot_token, "Bot token not set in bot instance")
        self.assertIsNotNone(self.bot.db, "Database manager not initialized")
        self.assertIsNotNone(self.bot.quiz_questions, "Quiz questions not loaded")
        self.assertIsNotNone(self.bot.lessons, "Lessons not loaded")
        
        print("✅ Configuration test passed!")
    
    def test_data_validation(self):
        """Test data validation and edge cases"""
        print("🧪 Testing data validation...")
        
        # Test invalid user creation
        invalid_user = User(
            user_id=0,  # Invalid user ID
            username="",
            first_name="",
            last_name="",
            language_code=""
        )
        
        # This should still work (bot should handle empty fields gracefully)
        result = self.db.create_user(invalid_user)
        self.assertTrue(result, "Should handle users with empty fields")
        
        # Test non-existent user retrieval
        non_existent = self.db.get_user(999999)
        self.assertIsNone(non_existent, "Should return None for non-existent user")
        
        print("✅ Data validation test passed!")

def test_bot_functionality():
    """Test basic bot functionality without Telegram API"""
    print("\n🤖 Testing Bot Functionality...")
    
    try:
        bot = EducationalBot()
        
        # Test quiz question filtering
        math_questions = [q for q in bot.quiz_questions if q.category == "Mathematics"]
        print(f"📊 Found {len(math_questions)} Mathematics questions")
        
        science_questions = [q for q in bot.quiz_questions if q.category == "Science"]
        print(f"🔬 Found {len(science_questions)} Science questions")
        
        # Test lesson filtering
        math_lessons = [l for l in bot.lessons if l.category == "Mathematics"]
        print(f"📚 Found {len(math_lessons)} Mathematics lessons")
        
        # Test difficulty levels
        beginner_questions = [q for q in bot.quiz_questions if q.difficulty == "beginner"]
        intermediate_questions = [q for q in bot.quiz_questions if q.difficulty == "intermediate"]
        advanced_questions = [q for q in bot.quiz_questions if q.difficulty == "advanced"]
        
        print(f"🎯 Difficulty distribution:")
        print(f"   Beginner: {len(beginner_questions)} questions")
        print(f"   Intermediate: {len(intermediate_questions)} questions")
        print(f"   Advanced: {len(advanced_questions)} questions")
        
        print("✅ Bot functionality test passed!")
        
    except Exception as e:
        print(f"❌ Bot functionality test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🎓 Educational Telegram Bot - Test Suite")
    print("=" * 50)
    
    # Test configuration first
    try:
        from config import Config
        print(f"🔑 Bot Token: {'✅ Configured' if Config.TELEGRAM_BOT_TOKEN else '❌ Missing'}")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    # Run additional functionality tests
    test_bot_functionality()
    
    print("\n🎉 All tests completed!")
    print("\n🚀 Ready to run the bot with: python telegram_education_bot.py")
    print("\n💡 Tips:")
    print("   • Make sure to set TELEGRAM_CHAT_ID environment variable if needed")
    print("   • Test the bot with a small group first")
    print("   • Monitor logs for any issues")
    print("   • Add more questions and lessons as needed")

if __name__ == "__main__":
    main()