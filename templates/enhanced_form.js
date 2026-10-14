// Enhanced Patient Form Manager
const PatientFormManager = {
    currentStep: 1,
    totalSteps: 5,
    formData: {},
    
    // Generate enhanced form with mandatory fields
    generateEnhancedForm() {
        return `
        <div class="intake-container">
            <!-- Progress Header -->
            <div class="progress-header card-premium" style="margin-bottom: 32px;">
                <div class="progress-title">
                    <h1 class="text-heading text-premium">
                        <i class="fas fa-clipboard-check"></i>
                        Complete Health Assessment
                    </h1>
                    <p class="text-subheading">Please fill all required fields for accurate medical consultation</p>
                </div>
                
                <div class="progress-container">
                    <div class="progress-steps">
                        <div class="step-indicator active" data-step="1">
                            <div class="step-circle"><i class="fas fa-user"></i></div>
                            <span class="step-label">Personal Info</span>
                        </div>
                        <div class="step-indicator" data-step="2">
                            <div class="step-circle"><i class="fas fa-weight"></i></div>
                            <span class="step-label">Physical</span>
                        </div>
                        <div class="step-indicator" data-step="3">
                            <div class="step-circle"><i class="fas fa-bed"></i></div>
                            <span class="step-label">Sleep & Lifestyle</span>
                        </div>
                        <div class="step-indicator" data-step="4">
                            <div class="step-circle"><i class="fas fa-heartbeat"></i></div>
                            <span class="step-label">Medical History</span>
                        </div>
                        <div class="step-indicator" data-step="5">
                            <div class="step-circle"><i class="fas fa-phone"></i></div>
                            <span class="step-label">Emergency Contact</span>
                        </div>
                    </div>
                    <div class="progress-line">
                        <div class="progress-fill" style="width: 20%;"></div>
                    </div>
                </div>
            </div>

            <!-- Step 1: Personal Information -->
            <div class="form-step active" id="step-1">
                <div class="step-card card-premium">
                    <div class="step-header">
                        <div class="step-icon">
                            <i class="fas fa-user"></i>
                        </div>
                        <div class="step-title">
                            <h2>Personal Information</h2>
                            <p>Basic details for your medical profile</p>
                        </div>
                    </div>

                    <div class="form-grid">
                        <div class="form-row">
                            <div class="premium-input-group">
                                <label class="premium-label field-required">
                                    <i class="fas fa-user-circle"></i>
                                    Full Name
                                </label>
                                <input type="text" class="premium-input required" id="fullName" name="fullName" 
                                    placeholder="Enter your full name" required>
                                <div class="input-border"></div>
                                <span class="error-message" id="fullName-error"></span>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="premium-input-group">
                                <label class="premium-label field-required">
                                    <i class="fas fa-calendar"></i>
                                    Date of Birth
                                </label>
                                <input type="date" class="premium-input required" id="dateOfBirth" name="dateOfBirth" required>
                                <div class="input-border"></div>
                                <span class="error-message" id="dateOfBirth-error"></span>
                                <div class="age-display" id="ageDisplay"></div>
                            </div>
                            <div class="premium-input-group">
                                <label class="premium-label field-required">
                                    <i class="fas fa-venus-mars"></i>
                                    Gender
                                </label>
                                <select class="premium-select required" id="gender" name="gender" required>
                                    <option value="">Select Gender</option>
                                    <option value="male">Male</option>
                                    <option value="female">Female</option>
                                    <option value="other">Other</option>
                                    <option value="prefer-not-to-say">Prefer not to say</option>
                                </select>
                                <div class="input-border"></div>
                                <span class="error-message" id="gender-error"></span>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="premium-input-group">
                                <label class="premium-label field-required">
                                    <i class="fas fa-envelope"></i>
                                    Email Address
                                </label>
                                <input type="email" class="premium-input required" id="email" name="email" 
                                    placeholder="your.email@example.com" required>
                                <div class="input-border"></div>
                                <span class="error-message" id="email-error"></span>
                            </div>
                            <div class="premium-input-group">
                                <label class="premium-label field-required">
                                    <i class="fas fa-phone"></i>
                                    Phone Number
                                </label>
                                <input type="tel" class="premium-input required" id="phone" name="phone" 
                                    placeholder="+1 (555) 123-4567" required>
                                <div class="input-border"></div>
                                <span class="error-message" id="phone-error"></span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Step 2: Physical Measurements -->
            <div class="form-step" id="step-2">
                <div class="step-card card-premium">
                    <div class="step-header">
                        <div class="step-icon">
                            <i class="fas fa-weight"></i>
                        </div>
                        <div class="step-title">
                            <h2>Physical Measurements</h2>
                            <p>Current measurements for accurate health assessment</p>
                        </div>
                    </div>

                    <div class="form-grid">
                        <div class="measurement-group">
                            <div class="measurement-card">
                                <h3 class="field-required"><i class="fas fa-arrows-alt-v"></i> Height</h3>
                                <div class="measurement-input">
                                    <input type="number" class="premium-input required" id="heightCm" name="height" 
                                        placeholder="170" min="50" max="250" required>
                                    <span class="unit">cm</span>
                                </div>
                                <div class="range-slider">
                                    <input type="range" id="heightRange" min="50" max="250" value="170">
                                </div>
                                <span class="error-message" id="height-error"></span>
                            </div>

                            <div class="measurement-card">
                                <h3 class="field-required"><i class="fas fa-weight-hanging"></i> Weight</h3>
                                <div class="measurement-input">
                                    <input type="number" class="premium-input required" id="weightKg" name="weight" 
                                        placeholder="70" min="20" max="200" required>
                                    <span class="unit">kg</span>
                                </div>
                                <div class="range-slider">
                                    <input type="range" id="weightRange" min="20" max="200" value="70">
                                </div>
                                <span class="error-message" id="weight-error"></span>
                            </div>
                        </div>

                        <!-- BMI Calculator Display -->
                        <div class="bmi-display card-premium" style="margin-top: 24px;">
                            <h3><i class="fas fa-calculator"></i> BMI Calculator</h3>
                            <div class="bmi-result">
                                <div class="bmi-value" id="bmiValue">--</div>
                                <div class="bmi-category" id="bmiCategory">Enter height and weight</div>
                            </div>
                        </div>

                        <div class="form-row" style="margin-top: 24px;">
                            <div class="premium-input-group">
                                <label class="premium-label">
                                    <i class="fas fa-tint"></i>
                                    Blood Type
                                </label>
                                <select class="premium-select" id="bloodType" name="bloodType">
                                    <option value="">Select Blood Type</option>
                                    <option value="A+">A+</option>
                                    <option value="A-">A-</option>
                                    <option value="B+">B+</option>
                                    <option value="B-">B-</option>
                                    <option value="AB+">AB+</option>
                                    <option value="AB-">AB-</option>
                                    <option value="O+">O+</option>
                                    <option value="O-">O-</option>
                                    <option value="unknown">Unknown</option>
                                </select>
                                <div class="input-border"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Step 3: Sleep & Lifestyle -->
            <div class="form-step" id="step-3">
                <div class="step-card card-premium">
                    <div class="step-header">
                        <div class="step-icon">
                            <i class="fas fa-bed"></i>
                        </div>
                        <div class="step-title">
                            <h2>Sleep & Lifestyle</h2>
                            <p>Daily habits and sleep patterns for better recommendations</p>
                        </div>
                    </div>

                    <div class="lifestyle-grid">
                        <div class="lifestyle-card">
                            <h4 class="field-required">
                                <i class="fas fa-moon"></i>
                                Sleep Information
                            </h4>
                            <div class="premium-input-group">
                                <label class="premium-label">Average Sleep Hours</label>
                                <input type="number" class="premium-input required" id="sleepHours" name="sleepHours" 
                                    placeholder="8" step="0.5" min="1" max="16" required>
                                <div class="input-border"></div>
                                <span class="error-message" id="sleepHours-error"></span>
                            </div>
                            <div class="premium-input-group" style="margin-top: 16px;">
                                <label class="premium-label">Sleep Quality</label>
                                <select class="premium-select required" id="sleepQuality" name="sleepQuality" required>
                                    <option value="">Rate your sleep</option>
                                    <option value="excellent">Excellent</option>
                                    <option value="good">Good</option>
                                    <option value="fair">Fair</option>
                                    <option value="poor">Poor</option>
                                    <option value="very-poor">Very Poor</option>
                                </select>
                                <div class="input-border"></div>
                                <span class="error-message" id="sleepQuality-error"></span>
                            </div>
                        </div>

                        <div class="lifestyle-card">
                            <h4 class="field-required">
                                <i class="fas fa-dumbbell"></i>
                                Exercise & Habits
                            </h4>
                            <div class="premium-input-group">
                                <label class="premium-label">Exercise Frequency</label>
                                <select class="premium-select required" id="exerciseFreq" name="exerciseFreq" required>
                                    <option value="">How often do you exercise?</option>
                                    <option value="daily">Daily</option>
                                    <option value="weekly-5-6">5-6 times per week</option>
                                    <option value="weekly-3-4">3-4 times per week</option>
                                    <option value="weekly-1-2">1-2 times per week</option>
                                    <option value="rarely">Rarely</option>
                                    <option value="never">Never</option>
                                </select>
                                <div class="input-border"></div>
                                <span class="error-message" id="exerciseFreq-error"></span>
                            </div>
                            <div class="premium-input-group" style="margin-top: 16px;">
                                <label class="premium-label">Smoking Status</label>
                                <select class="premium-select required" id="smokingStatus" name="smokingStatus" required>
                                    <option value="">Select smoking status</option>
                                    <option value="never">Never smoked</option>
                                    <option value="former">Former smoker</option>
                                    <option value="current">Current smoker</option>
                                    <option value="occasional">Occasional smoker</option>
                                </select>
                                <div class="input-border"></div>
                                <span class="error-message" id="smokingStatus-error"></span>
                            </div>
                        </div>

                        <div class="lifestyle-card">
                            <h4 class="field-required">
                                <i class="fas fa-wine-glass-alt"></i>
                                Alcohol & Stress
                            </h4>
                            <div class="premium-input-group">
                                <label class="premium-label">Alcohol Consumption</label>
                                <select class="premium-select required" id="alcoholConsumption" name="alcoholConsumption" required>
                                    <option value="">Select alcohol use</option>
                                    <option value="never">Never</option>
                                    <option value="rarely">Rarely</option>
                                    <option value="weekly">Weekly (1-3 drinks)</option>
                                    <option value="moderate">Moderate (4-7 drinks)</option>
                                    <option value="heavy">Heavy (8+ drinks)</option>
                                </select>
                                <div class="input-border"></div>
                                <span class="error-message" id="alcoholConsumption-error"></span>
                            </div>
                            <div class="premium-input-group" style="margin-top: 16px;">
                                <label class="premium-label">Stress Level</label>
                                <select class="premium-select required" id="stressLevel" name="stressLevel" required>
                                    <option value="">Rate your stress</option>
                                    <option value="minimal">Minimal</option>
                                    <option value="low">Low</option>
                                    <option value="moderate">Moderate</option>
                                    <option value="high">High</option>
                                    <option value="extreme">Extreme</option>
                                </select>
                                <div class="input-border"></div>
                                <span class="error-message" id="stressLevel-error"></span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Step 4: Medical History -->
            <div class="form-step" id="step-4">
                <div class="step-card card-premium">
                    <div class="step-header">
                        <div class="step-icon">
                            <i class="fas fa-stethoscope"></i>
                        </div>
                        <div class="step-title">
                            <h2>Medical History</h2>
                            <p>Important medical information for personalized care</p>
                        </div>
                    </div>

                    <div class="form-grid">
                        <div class="premium-input-group">
                            <label class="premium-label">
                                <i class="fas fa-clipboard-list"></i>
                                Current Medical Conditions
                            </label>
                            <div class="checkbox-group" style="margin-top: 12px;">
                                <label class="checkbox-item">
                                    <input type="checkbox" name="conditions" value="diabetes">
                                    <span class="checkmark"></span>
                                    Diabetes
                                </label>
                                <label class="checkbox-item">
                                    <input type="checkbox" name="conditions" value="hypertension">
                                    <span class="checkmark"></span>
                                    High Blood Pressure
                                </label>
                                <label class="checkbox-item">
                                    <input type="checkbox" name="conditions" value="heart-disease">
                                    <span class="checkmark"></span>
                                    Heart Disease
                                </label>
                                <label class="checkbox-item">
                                    <input type="checkbox" name="conditions" value="asthma">
                                    <span class="checkmark"></span>
                                    Asthma
                                </label>
                                <label class="checkbox-item">
                                    <input type="checkbox" name="conditions" value="arthritis">
                                    <span class="checkmark"></span>
                                    Arthritis
                                </label>
                                <label class="checkbox-item">
                                    <input type="checkbox" name="conditions" value="none">
                                    <span class="checkmark"></span>
                                    None of the above
                                </label>
                            </div>
                        </div>

                        <div class="premium-input-group" style="margin-top: 24px;">
                            <label class="premium-label">
                                <i class="fas fa-pills"></i>
                                Current Medications
                            </label>
                            <textarea class="premium-input" id="medications" name="medications" rows="3"
                                    placeholder="List your current medications, dosages, and frequency. If none, write 'None'."></textarea>
                            <div class="input-border"></div>
                        </div>

                        <div class="premium-input-group" style="margin-top: 24px;">
                            <label class="premium-label field-required">
                                <i class="fas fa-exclamation-triangle"></i>
                                Allergies
                            </label>
                            <textarea class="premium-input required" id="allergies" name="allergies" rows="3" required
                                    placeholder="List any drug allergies, food allergies, or environmental allergies. If none, write 'None'."></textarea>
                            <div class="input-border"></div>
                            <span class="error-message" id="allergies-error"></span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Step 5: Emergency Contact -->
            <div class="form-step" id="step-5">
                <div class="step-card card-premium">
                    <div class="step-header">
                        <div class="step-icon">
                            <i class="fas fa-phone-alt"></i>
                        </div>
                        <div class="step-title">
                            <h2>Emergency Contact</h2>
                            <p>Someone we can contact in case of medical emergency</p>
                        </div>
                    </div>

                    <div class="form-grid">
                        <div class="premium-input-group">
                            <label class="premium-label field-required">
                                <i class="fas fa-user-shield"></i>
                                Emergency Contact Name
                            </label>
                            <input type="text" class="premium-input required" id="emergencyName" name="emergencyName" 
                                placeholder="Full name of emergency contact" required>
                            <div class="input-border"></div>
                            <span class="error-message" id="emergencyName-error"></span>
                        </div>
                        <div class="form-row" style="margin-top: 20px;">
                            <div class="premium-input-group">
                                <label class="premium-label field-required">
                                    <i class="fas fa-phone"></i>
                                    Emergency Contact Phone
                                </label>
                                <input type="tel" class="premium-input required" id="emergencyPhone" name="emergencyPhone" 
                                    placeholder="+1 (555) 123-4567" required>
                                <div class="input-border"></div>
                                <span class="error-message" id="emergencyPhone-error"></span>
                            </div>
                            <div class="premium-input-group">
                                <label class="premium-label field-required">
                                    <i class="fas fa-heart"></i>
                                    Relationship
                                </label>
                                <select class="premium-select required" id="emergencyRelation" name="emergencyRelation" required>
                                    <option value="">Select relationship</option>
                                    <option value="spouse">Spouse/Partner</option>
                                    <option value="parent">Parent</option>
                                    <option value="sibling">Sibling</option>
                                    <option value="child">Child</option>
                                    <option value="friend">Friend</option>
                                    <option value="other">Other</option>
                                </select>
                                <div class="input-border"></div>
                                <span class="error-message" id="emergencyRelation-error"></span>
                            </div>
                        </div>

                        <div class="consent-section" style="margin-top: 32px; padding: 20px; background: rgba(0, 102, 255, 0.05); border-radius: 12px; border: 1px solid rgba(0, 102, 255, 0.1);">
                            <label class="checkbox-item large">
                                <input type="checkbox" id="dataConsent" name="dataConsent" required>
                                <span class="checkmark"></span>
                                <span class="consent-text">
                                    <strong>Data Consent:</strong> I consent to the collection and processing of my medical data for AI-powered health recommendations. 
                                    This data will be stored securely and used only for providing personalized medical guidance.
                                </span>
                            </label>
                            <span class="error-message" id="dataConsent-error"></span>
                        </div>

                        <div class="completion-notice" style="text-align: center; margin-top: 24px; padding: 20px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(0, 102, 255, 0.05) 100%); border-radius: 12px;">
                            <i class="fas fa-check-circle" style="font-size: 48px; color: var(--success-green); margin-bottom: 12px;"></i>
                            <h3 style="color: var(--gray-900); margin-bottom: 8px;">Ready for Medical Consultation</h3>
                            <p style="color: var(--gray-600); margin: 0;">Your comprehensive health profile will help our AI provide accurate and personalized medical recommendations.</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Enhanced Navigation -->
            <div class="form-navigation" style="margin-top: 40px; display: flex; justify-content: space-between; align-items: center;">
                <button type="button" class="btn-secondary" id="prevBtn" onclick="PatientFormManager.previousStep()" disabled>
                    <i class="fas fa-chevron-left"></i>
                    Previous
                </button>
                
                <div class="step-dots" style="display: flex; gap: 8px;">
                    <span class="step-dot active" data-step="1"></span>
                    <span class="step-dot" data-step="2"></span>
                    <span class="step-dot" data-step="3"></span>
                    <span class="step-dot" data-step="4"></span>
                    <span class="step-dot" data-step="5"></span>
                </div>
                
                <button type="button" class="btn-premium" id="nextBtn" onclick="PatientFormManager.nextStep()">
                    Next
                    <i class="fas fa-chevron-right"></i>
                </button>
                <button type="submit" class="btn-premium hidden" id="submitBtn" onclick="PatientFormManager.submitForm()">
                    <i class="fas fa-user-md"></i>
                    Start Medical Consultation
                </button>
            </div>
        </div>
        `;
    },

    // Validation functions
    validateStep(step) {
        const stepElement = document.getElementById(`step-${step}`);
        const requiredInputs = stepElement.querySelectorAll('.required');
        let isValid = true;

        requiredInputs.forEach(input => {
            if (!input.value.trim()) {
                this.showError(input, 'This field is required');
                isValid = false;
            } else {
                this.clearError(input);
            }
        });

        return isValid;
    },

    showError(input, message) {
        input.classList.add('error');
        const errorElement = document.getElementById(`${input.name || input.id}-error`);
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.classList.add('show');
        }
    },

    clearError(input) {
        input.classList.remove('error');
        input.classList.add('valid');
        const errorElement = document.getElementById(`${input.name || input.id}-error`);
        if (errorElement) {
            errorElement.classList.remove('show');
        }
    },

    nextStep() {
        if (this.validateStep(this.currentStep)) {
            if (this.currentStep < this.totalSteps) {
                this.currentStep++;
                this.showStep(this.currentStep);
                this.updateProgress();
            } else {
                this.showSubmitButton();
            }
        }
    },

    previousStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this.showStep(this.currentStep);
            this.updateProgress();
        }
    },

    showStep(step) {
        // Hide all steps
        document.querySelectorAll('.form-step').forEach(stepEl => {
            stepEl.classList.remove('active');
        });
        
        // Show current step
        document.getElementById(`step-${step}`).classList.add('active');
        
        // Update navigation buttons
        document.getElementById('prevBtn').disabled = step === 1;
        document.getElementById('nextBtn').style.display = step === this.totalSteps ? 'none' : 'inline-flex';
        document.getElementById('submitBtn').style.display = step === this.totalSteps ? 'inline-flex' : 'none';
        
        // Update step indicators
        document.querySelectorAll('.step-dot').forEach((dot, index) => {
            dot.classList.toggle('active', index + 1 === step);
            dot.classList.toggle('completed', index + 1 < step);
        });
    },

    updateProgress() {
        const progressPercent = (this.currentStep / this.totalSteps) * 100;
        document.querySelector('.progress-fill').style.width = progressPercent + '%';
        
        document.querySelectorAll('.step-indicator').forEach((indicator, index) => {
            indicator.classList.toggle('active', index + 1 === this.currentStep);
            indicator.classList.toggle('completed', index + 1 < this.currentStep);
        });
    },

    async submitForm() {
        if (this.validateStep(this.currentStep)) {
            // Collect all form data
            const formData = new FormData(document.querySelector('.intake-container'));
            const patientData = Object.fromEntries(formData.entries());
            
            // Add checkbox data
            document.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => {
                const name = cb.name;
                if (!patientData[name]) patientData[name] = [];
                if (Array.isArray(patientData[name])) {
                    patientData[name].push(cb.value);
                } else {
                    patientData[name] = [patientData[name], cb.value];
                }
            });

            try {
                // Submit to backend
                const response = await fetch('/api/patient/profile', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(patientData)
                });

                if (response.ok) {
                    // Mark intake as completed
                    localStorage.setItem('medai_intake_completed', 'true');
                    localStorage.setItem('medai_patient_profile', JSON.stringify(patientData));
                    
                    // Redirect to chat
                    window.MedAI.showChatInterface();
                    window.MedAI.showSuccessMessage('Health profile completed successfully! You can now get personalized medical recommendations.');
                } else {
                    throw new Error('Failed to save patient profile');
                }
            } catch (error) {
                console.error('Error submitting form:', error);
                alert('Error saving your profile. Please try again.');
            }
        }
    },

    // Initialize enhanced functionality
    init() {
        this.setupRealTimeValidation();
        this.setupBMICalculator();
        this.setupAgeCalculator();
        this.setupRangeSliders();
    },

    setupRealTimeValidation() {
        document.querySelectorAll('.required').forEach(input => {
            input.addEventListener('blur', () => {
                if (input.value.trim()) {
                    this.clearError(input);
                } else {
                    this.showError(input, 'This field is required');
                }
            });
        });
    },

    setupBMICalculator() {
        const heightInput = document.getElementById('heightCm');
        const weightInput = document.getElementById('weightKg');
        const bmiValue = document.getElementById('bmiValue');
        const bmiCategory = document.getElementById('bmiCategory');

        function calculateBMI() {
            const height = parseFloat(heightInput.value) / 100; // Convert to meters
            const weight = parseFloat(weightInput.value);
            
            if (height > 0 && weight > 0) {
                const bmi = (weight / (height * height)).toFixed(1);
                bmiValue.textContent = bmi;
                
                let category = '';
                if (bmi < 18.5) category = 'Underweight';
                else if (bmi < 25) category = 'Normal';
                else if (bmi < 30) category = 'Overweight';
                else category = 'Obese';
                
                bmiCategory.textContent = category;
                bmiCategory.className = 'bmi-category ' + category.toLowerCase();
            }
        }

        heightInput?.addEventListener('input', calculateBMI);
        weightInput?.addEventListener('input', calculateBMI);
    },

    setupAgeCalculator() {
        const dobInput = document.getElementById('dateOfBirth');
        const ageDisplay = document.getElementById('ageDisplay');

        dobInput?.addEventListener('change', () => {
            const birthDate = new Date(dobInput.value);
            const today = new Date();
            const age = Math.floor((today - birthDate) / (365.25 * 24 * 60 * 60 * 1000));
            
            if (age >= 0 && age <= 150) {
                ageDisplay.textContent = `Age: ${age} years`;
                ageDisplay.style.color = 'var(--success-green)';
            } else {
                ageDisplay.textContent = 'Please enter a valid date';
                ageDisplay.style.color = 'var(--danger-red)';
            }
        });
    },

    setupRangeSliders() {
        const heightRange = document.getElementById('heightRange');
        const weightRange = document.getElementById('weightRange');
        const heightInput = document.getElementById('heightCm');
        const weightInput = document.getElementById('weightKg');

        heightRange?.addEventListener('input', () => {
            heightInput.value = heightRange.value;
            heightInput.dispatchEvent(new Event('input'));
        });

        weightRange?.addEventListener('input', () => {
            weightInput.value = weightRange.value;
            weightInput.dispatchEvent(new Event('input'));
        });

        heightInput?.addEventListener('input', () => {
            if (heightRange) heightRange.value = heightInput.value;
        });

        weightInput?.addEventListener('input', () => {
            if (weightRange) weightRange.value = weightInput.value;
        });
    }
};

// Export for use in main application
window.PatientFormManager = PatientFormManager;