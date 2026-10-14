
// Enhanced MedBot Frontend Integration
class MedBotEnhanced {
    constructor() {
        this.baseUrl = window.location.origin;
        this.formCompleted = false;
        this.userProfile = null;
    }
    
    async checkFormCompletion() {
        try {
            const response = await fetch(`${this.baseUrl}/api/check-form-completion`);
            const data = await response.json();
            
            this.formCompleted = data.form_completed;
            this.userProfile = data;
            
            if (data.form_completed) {
                this.showChatInterface();
                await this.loadPersonalizedGreeting();
            } else {
                this.showPatientForm();
            }
            
            return data;
        } catch (error) {
            console.error('Form completion check failed:', error);
            this.showPatientForm();
        }
    }
    
    async loadPersonalizedGreeting() {
        try {
            const response = await fetch(`${this.baseUrl}/api/get-personalized-greeting`);
            const greeting = await response.json();
            
            this.displayGreeting(greeting);
        } catch (error) {
            console.error('Greeting load failed:', error);
            this.displayDefaultGreeting();
        }
    }
    
    displayGreeting(greeting) {
        const chatContainer = document.getElementById('chat-container');
        if (!chatContainer) return;
        
        const greetingHTML = `
            <div class="ai-greeting-message">
                <div class="greeting-header">
                    <h3>${greeting.greeting}</h3>
                    <p>${greeting.personalized_message}</p>
                </div>
                <div class="capabilities-list">
                    <h4>How I can help you:</h4>
                    <ul>
                        ${greeting.capabilities.map(cap => `<li>${cap}</li>`).join('')}
                    </ul>
                </div>
                <div class="disclaimer">
                    <small>${greeting.disclaimer}</small>
                </div>
            </div>
        `;
        
        chatContainer.innerHTML = greetingHTML + chatContainer.innerHTML;
    }
    
    displayDefaultGreeting() {
        const greeting = {
            greeting: "Hello! I'm your AI Medical Assistant.",
            personalized_message: "I'm here to help with your medical questions.",
            capabilities: ["Medical information", "Health guidance", "Symptom analysis"],
            disclaimer: "Consult healthcare providers for serious concerns."
        };
        
        this.displayGreeting(greeting);
    }
    
    showChatInterface() {
        // Hide form, show chat
        const formContainer = document.getElementById('patient-form-container');
        const chatContainer = document.getElementById('chat-interface-container');
        
        if (formContainer) formContainer.style.display = 'none';
        if (chatContainer) chatContainer.style.display = 'block';
    }
    
    showPatientForm() {
        // Show form, hide chat
        const formContainer = document.getElementById('patient-form-container');
        const chatContainer = document.getElementById('chat-interface-container');
        
        if (formContainer) formContainer.style.display = 'block';
        if (chatContainer) chatContainer.style.display = 'none';
    }
    
    async submitForm(formData) {
        try {
            const response = await fetch(`${this.baseUrl}/api/patient/profile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.formCompleted = true;
                this.showChatInterface();
                await this.loadPersonalizedGreeting();
                return true;
            } else {
                throw new Error(result.error || 'Form submission failed');
            }
        } catch (error) {
            console.error('Form submission failed:', error);
            throw error;
        }
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.medBot = new MedBotEnhanced();
    window.medBot.checkFormCompletion();
});
