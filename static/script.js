// COSMIC AI Binary Signal Bot - Frontend JavaScript

class CosmicAI {
    constructor() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.uploadPreview = document.getElementById('uploadPreview');
        this.previewImage = document.getElementById('previewImage');
        this.removeBtn = document.getElementById('removeBtn');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.resultsSection = document.getElementById('resultsSection');
        
        this.selectedFile = null;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Drag and drop events
        this.uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadArea.addEventListener('drop', (e) => this.handleDrop(e));
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        
        // Remove button
        this.removeBtn.addEventListener('click', (e) => this.removeFile(e));
        
        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeChart());
        
        // Prevent default drag behaviors on document
        document.addEventListener('dragover', (e) => e.preventDefault());
        document.addEventListener('drop', (e) => e.preventDefault());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('drag-over');
    }

    handleDragLeave(e) {
        e.preventDefault();
        if (!this.uploadArea.contains(e.relatedTarget)) {
            this.uploadArea.classList.remove('drag-over');
        }
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file (JPG, PNG, WEBP)');
            return;
        }

        // Validate file size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            this.showError('File size must be less than 16MB');
            return;
        }

        this.selectedFile = file;
        this.displayPreview(file);
        this.enableAnalyzeButton();
    }

    displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.uploadPreview.style.display = 'block';
            this.uploadArea.querySelector('.upload-content').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    removeFile(e) {
        e.stopPropagation();
        this.selectedFile = null;
        this.uploadPreview.style.display = 'none';
        this.uploadArea.querySelector('.upload-content').style.display = 'block';
        this.fileInput.value = '';
        this.disableAnalyzeButton();
        this.hideResults();
    }

    enableAnalyzeButton() {
        this.analyzeBtn.disabled = false;
        this.analyzeBtn.style.opacity = '1';
    }

    disableAnalyzeButton() {
        this.analyzeBtn.disabled = true;
        this.analyzeBtn.style.opacity = '0.6';
    }

    async analyzeChart() {
        if (!this.selectedFile) {
            this.showError('Please select a chart image first');
            return;
        }

        this.showLoading();
        
        try {
            const formData = new FormData();
            formData.append('chart', this.selectedFile);

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.hideLoading();
            this.displayResults(result);

        } catch (error) {
            console.error('Analysis error:', error);
            this.hideLoading();
            this.showError('Failed to analyze chart. Please try again.');
        }
    }

    displayResults(result) {
        // Show results section
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Update timestamp
        const timestamp = new Date(result.timestamp).toLocaleString();
        document.getElementById('resultTimestamp').textContent = timestamp;

        // Update signal display
        this.updateSignalDisplay(result);

        // Update confidence meter
        this.updateConfidenceMeter(result.confidence);

        // Update analysis details
        this.updateAnalysisDetails(result);

        // Show Telegram status if signal was sent
        this.updateTelegramStatus(result);
    }

    updateSignalDisplay(result) {
        const signalIcon = document.getElementById('signalIcon');
        const signalText = document.getElementById('signalText');

        // Clear previous classes
        signalIcon.className = 'signal-icon';
        signalText.className = 'signal-text';

        const signal = result.signal.toLowerCase().replace(' ', '-');
        
        // Set icon and text based on signal
        switch (signal) {
            case 'call':
                signalIcon.innerHTML = '<i class="fas fa-arrow-up"></i>';
                signalIcon.classList.add('call');
                signalText.classList.add('call');
                signalText.textContent = 'CALL';
                break;
            case 'put':
                signalIcon.innerHTML = '<i class="fas fa-arrow-down"></i>';
                signalIcon.classList.add('put');
                signalText.classList.add('put');
                signalText.textContent = 'PUT';
                break;
            default:
                signalIcon.innerHTML = '<i class="fas fa-pause"></i>';
                signalIcon.classList.add('no-trade');
                signalText.classList.add('no-trade');
                signalText.textContent = 'NO TRADE';
        }
    }

    updateConfidenceMeter(confidence) {
        const confidenceFill = document.getElementById('confidenceFill');
        const confidenceValue = document.getElementById('confidenceValue');

        // Animate confidence fill
        setTimeout(() => {
            confidenceFill.style.width = `${confidence}%`;
        }, 100);

        confidenceValue.textContent = `${confidence}%`;
    }

    updateAnalysisDetails(result) {
        const reasonBox = document.getElementById('reasonBox');
        const detailsGrid = document.getElementById('detailsGrid');

        // Update reason
        reasonBox.textContent = result.reason;

        // Clear and update details grid
        detailsGrid.innerHTML = '';

        if (result.analysis_details) {
            const details = result.analysis_details;

            // Add momentum info
            if (details.momentum) {
                this.addDetailItem(detailsGrid, 'Momentum', 
                    `${details.momentum.direction} (${details.momentum.strength.toFixed(1)}%)`);
            }

            // Add trend info
            if (details.trend) {
                this.addDetailItem(detailsGrid, 'Trend', 
                    `${details.trend.direction} (${details.trend.strength.toFixed(1)}%)`);
            }

            // Add patterns
            if (details.patterns && details.patterns.length > 0) {
                const patterns = details.patterns.map(p => p.replace('_', ' ')).join(', ');
                this.addDetailItem(detailsGrid, 'Patterns', patterns);
            }

            // Add psychology
            if (details.psychology) {
                this.addDetailItem(detailsGrid, 'Market Sentiment', 
                    `${details.psychology.sentiment} (${details.psychology.conviction} conviction)`);
            }

            // Add support/resistance
            if (details.support_resistance) {
                this.addDetailItem(detailsGrid, 'Price Position', 
                    details.support_resistance.current_position.replace('_', ' '));
            }
        }
    }

    addDetailItem(container, label, value) {
        const item = document.createElement('div');
        item.className = 'detail-item';
        item.innerHTML = `
            <div class="detail-label">${label}</div>
            <div class="detail-value">${value}</div>
        `;
        container.appendChild(item);
    }

    updateTelegramStatus(result) {
        const telegramStatus = document.getElementById('telegramStatus');
        
        // Show status if signal was sent (confidence above threshold)
        if (result.confidence >= 85 && result.signal !== 'NO TRADE') {
            telegramStatus.style.display = 'flex';
        } else {
            telegramStatus.style.display = 'none';
        }
    }

    showLoading() {
        this.loadingOverlay.style.display = 'flex';
        this.analyzeBtn.classList.add('loading');
        this.analyzeBtn.disabled = true;
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
        this.analyzeBtn.classList.remove('loading');
        this.analyzeBtn.disabled = false;
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
    }

    showError(message) {
        // Create and show error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.innerHTML = `
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;

        // Add error styles if not already present
        if (!document.querySelector('.error-notification-styles')) {
            const style = document.createElement('style');
            style.className = 'error-notification-styles';
            style.textContent = `
                .error-notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: var(--cosmic-danger);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    z-index: 1001;
                    max-width: 400px;
                    box-shadow: 0 8px 25px rgba(225, 112, 85, 0.3);
                    animation: slideIn 0.3s ease;
                }
                
                .error-notification button {
                    background: none;
                    border: none;
                    color: white;
                    cursor: pointer;
                    padding: 5px;
                    margin-left: auto;
                }
                
                @keyframes slideIn {
                    from { transform: translateX(100%); }
                    to { transform: translateX(0); }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(errorDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CosmicAI();
    
    // Add some visual enhancements
    addVisualEffects();
});

function addVisualEffects() {
    // Add smooth scrolling
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Add intersection observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe feature cards for animation
    document.querySelectorAll('.feature-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
}