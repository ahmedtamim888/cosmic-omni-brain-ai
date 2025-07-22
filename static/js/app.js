// 🔥 GHOST TRANSCENDENCE CORE ∞ vX - Frontend Logic

class GhostTranscendenceCore {
    constructor() {
        this.version = "∞ vX";
        this.isAnalyzing = false;
        this.currentFile = null;
        this.initializeEventListeners();
        this.showWelcomeAnimation();
    }

    initializeEventListeners() {
        // File input handling
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        
        // Upload area click
        uploadArea.addEventListener('click', () => {
            if (!this.isAnalyzing) {
                fileInput.click();
            }
        });
    }

    showWelcomeAnimation() {
        // Add subtle entrance animations
        const elements = document.querySelectorAll('.feature-card, .upload-area');
        elements.forEach((el, index) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                el.style.transition = 'all 0.6s ease';
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, index * 100);
        });
    }

    handleDragOver(e) {
        e.preventDefault();
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file (PNG, JPG, JPEG)');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size too large. Please select an image under 10MB.');
            return;
        }

        this.currentFile = file;
        this.startAnalysis();
    }

    async startAnalysis() {
        if (this.isAnalyzing) return;
        
        this.isAnalyzing = true;
        this.showAnalysisInterface();
        
        try {
            const result = await this.uploadAndAnalyze();
            this.showResults(result);
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.isAnalyzing = false;
        }
    }

    showAnalysisInterface() {
        // Hide upload section
        document.querySelector('.upload-section').style.display = 'none';
        
        // Show analysis section
        const analysisSection = document.getElementById('analysisSection');
        analysisSection.style.display = 'block';
        
        // Show loading overlay
        document.getElementById('loadingOverlay').style.display = 'flex';
        
        // Start progress simulation
        this.simulateProgress();
    }

    simulateProgress() {
        const steps = ['step1', 'step2', 'step3', 'step4'];
        const progressFill = document.getElementById('progressFill');
        let currentStep = 0;
        
        const progressInterval = setInterval(() => {
            // Update progress bar
            const progress = ((currentStep + 1) / steps.length) * 100;
            progressFill.style.width = progress + '%';
            
            // Update step indicators
            if (currentStep > 0) {
                document.getElementById(steps[currentStep - 1]).classList.remove('active');
            }
            
            if (currentStep < steps.length) {
                document.getElementById(steps[currentStep]).classList.add('active');
                currentStep++;
            }
            
            if (currentStep >= steps.length) {
                clearInterval(progressInterval);
            }
        }, 1000);
        
        // Store interval for cleanup
        this.progressInterval = progressInterval;
    }

    async uploadAndAnalyze() {
        const formData = new FormData();
        formData.append('chart_image', this.currentFile);
        
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }

    showResults(result) {
        // Hide loading overlay
        document.getElementById('loadingOverlay').style.display = 'none';
        
        // Hide analysis section
        document.getElementById('analysisSection').style.display = 'none';
        
        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
        
        // Populate results
        this.populateSignalData(result);
        this.populateReasoningData(result);
        this.populateMarketData(result);
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Add entrance animation
        resultsSection.style.opacity = '0';
        resultsSection.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            resultsSection.style.transition = 'all 0.8s ease';
            resultsSection.style.opacity = '1';
            resultsSection.style.transform = 'translateY(0)';
        }, 100);
    }

    populateSignalData(result) {
        const signal = result.signal || 'NO SIGNAL';
        const confidence = result.confidence || 0;
        
        // Update signal badge
        const signalBadge = document.getElementById('signalBadge');
        signalBadge.textContent = signal;
        signalBadge.className = 'signal-badge ' + signal.toLowerCase().replace(' ', '-');
        
        // Update signal type
        const signalType = document.getElementById('signalType');
        signalType.textContent = signal;
        signalType.className = 'signal-type ' + signal.toLowerCase().replace(' ', '-');
        
        // Update confidence
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceValue = document.getElementById('confidenceValue');
        
        setTimeout(() => {
            confidenceFill.style.width = confidence + '%';
            confidenceValue.textContent = confidence.toFixed(1) + '%';
        }, 500);
        
        // Update details
        document.getElementById('timeframe').textContent = result.timeframe || '1M';
        document.getElementById('target').textContent = result.time_target || 'Next candle';
        document.getElementById('strategy').textContent = this.formatStrategyName(result.strategy_type);
    }

    populateReasoningData(result) {
        const reasoningContent = document.getElementById('reasoningContent');
        reasoningContent.textContent = result.reasoning || 'Advanced AI analysis completed.';
    }

    populateMarketData(result) {
        // Market conditions
        const marketConditions = document.getElementById('marketConditions');
        const conditions = result.market_conditions || {};
        
        marketConditions.innerHTML = '';
        Object.entries(conditions).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'condition-item';
            item.innerHTML = `
                <span>${this.formatLabel(key)}</span>
                <span class="value ${this.getStatusClass(value)}">${value}</span>
            `;
            marketConditions.appendChild(item);
        });
        
        // Risk assessment
        const riskAssessment = document.getElementById('riskAssessment');
        const risk = result.risk_assessment || {};
        
        riskAssessment.innerHTML = '';
        Object.entries(risk).forEach(([key, value]) => {
            const item = document.createElement('div');
            item.className = 'risk-item';
            item.innerHTML = `
                <span>${this.formatLabel(key)}</span>
                <span class="value ${this.getRiskClass(key, value)}">${this.formatValue(value)}</span>
            `;
            riskAssessment.appendChild(item);
        });
        
        // Ghost factor
        const ghostFactor = document.getElementById('ghostFactor');
        const ghostValue = result.ghost_factor || 0.5;
        
        ghostFactor.innerHTML = `
            <div class="ghost-item">
                <span>Transcendence Level</span>
                <span class="value text-success">${(ghostValue * 100).toFixed(1)}%</span>
            </div>
            <div class="ghost-item">
                <span>Manipulation Immunity</span>
                <span class="value text-success">${ghostValue > 0.8 ? 'Maximum' : ghostValue > 0.6 ? 'High' : 'Medium'}</span>
            </div>
            <div class="ghost-item">
                <span>AI Evolution</span>
                <span class="value text-info">Active</span>
            </div>
        `;
    }

    formatStrategyName(strategy) {
        if (!strategy) return 'Adaptive Strategy';
        return strategy.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    formatLabel(key) {
        return key.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    formatValue(value) {
        if (typeof value === 'number') {
            return value.toFixed(2);
        }
        return value;
    }

    getStatusClass(value) {
        const lowerValue = String(value).toLowerCase();
        if (lowerValue.includes('bullish') || lowerValue.includes('up')) {
            return 'text-success';
        } else if (lowerValue.includes('bearish') || lowerValue.includes('down')) {
            return 'text-danger';
        } else if (lowerValue.includes('high')) {
            return 'text-warning';
        }
        return '';
    }

    getRiskClass(key, value) {
        if (key === 'level') {
            switch (value) {
                case 'low': return 'text-success';
                case 'medium': return 'text-warning';
                case 'high': return 'text-danger';
                default: return '';
            }
        }
        return '';
    }

    showError(message) {
        // Hide loading overlay
        document.getElementById('loadingOverlay').style.display = 'none';
        
        // Show error notification
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            z-index: 2000;
            max-width: 400px;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        `;
        
        errorDiv.innerHTML = `
            <div style="display: flex; align-items: center; gap: 10px;">
                <i class="fas fa-exclamation-triangle" style="font-size: 1.2rem;"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="
                    background: none;
                    border: none;
                    color: white;
                    cursor: pointer;
                    margin-left: auto;
                    font-size: 1.2rem;
                ">×</button>
            </div>
        `;
        
        document.body.appendChild(errorDiv);
        
        // Animate in
        setTimeout(() => {
            errorDiv.style.transform = 'translateX(0)';
        }, 100);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.style.transform = 'translateX(100%)';
                setTimeout(() => errorDiv.remove(), 300);
            }
        }, 5000);
        
        // Reset interface
        this.resetInterface();
    }

    resetInterface() {
        this.isAnalyzing = false;
        
        // Clear progress interval
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        // Reset progress
        document.getElementById('progressFill').style.width = '0%';
        
        // Reset steps
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active');
        });
        document.getElementById('step1').classList.add('active');
        
        // Show upload section
        document.querySelector('.upload-section').style.display = 'block';
        
        // Hide other sections
        document.getElementById('analysisSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
    }
}

// Global functions for button clicks
function analyzeNew() {
    ghostCore.resetInterface();
    
    // Clear file input
    document.getElementById('fileInput').value = '';
    ghostCore.currentFile = null;
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function downloadReport() {
    // Create a simple report download
    const reportData = {
        timestamp: new Date().toISOString(),
        version: "GHOST TRANSCENDENCE CORE ∞ vX",
        analysis: "Trading signal analysis report",
        // Add more report data as needed
    };
    
    const blob = new Blob([JSON.stringify(reportData, null, 2)], { 
        type: 'application/json' 
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ghost_analysis_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize the application
let ghostCore;

document.addEventListener('DOMContentLoaded', () => {
    ghostCore = new GhostTranscendenceCore();
    
    // Add some Easter eggs and advanced features
    initializeEasterEggs();
});

function initializeEasterEggs() {
    // Konami code for developer mode
    let konamiCode = '';
    const konamiSequence = 'ArrowUpArrowUpArrowDownArrowDownArrowLeftArrowRightArrowLeftArrowRightKeyBKeyA';
    
    document.addEventListener('keydown', (e) => {
        konamiCode += e.code;
        if (konamiCode.length > konamiSequence.length) {
            konamiCode = konamiCode.slice(-konamiSequence.length);
        }
        
        if (konamiCode === konamiSequence) {
            activateDeveloperMode();
        }
    });
    
    // Ghost mode activation on triple-click
    let clickCount = 0;
    document.querySelector('.logo i').addEventListener('click', () => {
        clickCount++;
        if (clickCount === 3) {
            activateGhostMode();
            clickCount = 0;
        }
        setTimeout(() => { clickCount = 0; }, 2000);
    });
}

function activateDeveloperMode() {
    console.log('🔥 GHOST TRANSCENDENCE CORE - Developer Mode Activated');
    
    // Add developer info to console
    console.log(`
    👻 GHOST TRANSCENDENCE CORE ∞ vX
    🧠 Ultimate AI Trading Bot
    ⚡ Manipulation Resistant
    🎯 Dynamic Strategy Creation
    
    Available in window.ghostCore
    `);
    
    // Make ghost core available globally for debugging
    window.ghostCore = ghostCore;
    
    // Show notification
    showNotification('🔥 Developer Mode Activated', 'Check console for advanced controls');
}

function activateGhostMode() {
    console.log('👻 Ghost Mode: Invisibility Protocol Activated');
    
    // Add subtle visual effects
    document.body.style.filter = 'hue-rotate(90deg)';
    setTimeout(() => {
        document.body.style.filter = '';
    }, 3000);
    
    showNotification('👻 Ghost Mode', 'Transcendence level increased');
}

function showNotification(title, message) {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 20px;
        background: linear-gradient(135deg, #00ff88, #00aaff);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        z-index: 2000;
        transform: translateX(-100%);
        transition: transform 0.3s ease;
    `;
    
    notification.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 5px;">${title}</div>
        <div style="opacity: 0.9;">${message}</div>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(-100%)';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Performance monitoring
window.addEventListener('load', () => {
    if (performance.mark) {
        performance.mark('ghost-core-loaded');
        console.log('👻 Ghost Transcendence Core loaded in:', performance.now().toFixed(2) + 'ms');
    }
});