#!/usr/bin/env python3
"""
Enhanced Chat Endpoints with Fixed Error Handling
=================================================
Fixes all the identified chat-related issues
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from flask import Flask, request, jsonify, session

# Add current directory to path
sys.path.insert(0, os.getcwd())

def create_enhanced_chat_routes(app: Flask, medical_chatbot=None, ai_system=None, user_manager=None):
    """Create enhanced chat routes with proper error handling"""
    
    @app.route('/api/chat/test', methods=['POST', 'GET'])
    def test_chat_endpoint():
        """Test endpoint to verify chat functionality"""
        try:
            if request.method == 'GET':
                return jsonify({
                    "status": "Chat endpoint is working",
                    "medical_chatbot_available": medical_chatbot is not None,
                    "ai_system_available": ai_system is not None,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Handle POST request
            if request.is_json:
                data = request.get_json()
                message = data.get('message', 'Test message')
            else:
                message = request.form.get('message', 'Test message')
            
            return jsonify({
                "response": f"Echo: {message}",
                "timestamp": datetime.now().isoformat(),
                "method": "test_endpoint"
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Test endpoint error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/api/chat/enhanced', methods=['POST'])
    def enhanced_chat_endpoint():
        """Enhanced chat endpoint with comprehensive error handling"""
        try:
            # Handle both JSON and form data
            if request.is_json:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                message = data.get('message', '').strip()
            else:
                message = request.form.get('message', '').strip()
            
            if not message:
                return jsonify({"error": "Message is required"}), 400
            
            # Get user context
            user_email = session.get('user_email', 'anonymous')
            user_id = session.get('user_id', 'anonymous')
            session_id = session.get('current_session_id', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            user_context = {
                'id': user_id,
                'email': user_email,
                'role': 'authenticated' if user_email != 'anonymous' else 'guest'
            }
            
            # Process through medical AI
            ai_response = "I'm MedAI, ready to help with your medical questions."
            processing_method = "fallback"
            
            # Try medical chatbot first
            if medical_chatbot:
                try:
                    # Handle async properly
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        ai_response = loop.run_until_complete(
                            medical_chatbot.process_query_with_context(message, user_context, session_id)
                        )
                        processing_method = "medical_chatbot"
                    finally:
                        loop.close()
                        
                except Exception as chatbot_error:
                    logging.error(f"Medical chatbot error: {chatbot_error}")
                    
                    # Fallback to AI system
                    if ai_system:
                        try:
                            ai_response = ai_system.process_intelligent_query_with_patient_context(
                                message, user_id, '', user_context
                            )
                            processing_method = "ai_system_fallback"
                        except Exception as ai_error:
                            logging.error(f"AI system fallback error: {ai_error}")
                            ai_response = f"I apologize, but I'm experiencing technical difficulties. Your message '{message[:50]}...' has been received. Please try again in a moment."
                            processing_method = "error_fallback"
            
            # Save chat if user is authenticated
            if user_email != 'anonymous' and user_manager:
                try:
                    user_manager.save_chat_message(user_email, message, ai_response, session_id)
                except Exception as save_error:
                    logging.warning(f"Chat save failed: {save_error}")
            
            return jsonify({
                "response": ai_response,
                "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_type": "authenticated" if user_email != 'anonymous' else "guest",
                "processing_method": processing_method,
                "medical_response": True
            })
            
        except Exception as e:
            logging.error(f"Enhanced chat endpoint error: {e}")
            return jsonify({
                "error": f"Chat processing failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/api/chat/simple', methods=['POST'])
    def simple_chat_endpoint():
        """Simple chat endpoint for testing"""
        try:
            # Get message from request
            if request.is_json:
                data = request.get_json() or {}
                message = data.get('message', '')
            else:
                message = request.form.get('message', '')
            
            if not message:
                return jsonify({"error": "Message required"}), 400
            
            # Simple response
            response = f"I received your message: '{message}'. This is a simple test response from MedAI."
            
            return jsonify({
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "endpoint": "simple"
            })
            
        except Exception as e:
            return jsonify({
                "error": f"Simple chat error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 500
    
    return app

if __name__ == "__main__":
    # Test the enhanced endpoints
    app = Flask(__name__)
    app.secret_key = 'test_secret_key'
    
    # Add the enhanced routes
    enhanced_app = create_enhanced_chat_routes(app)
    
    print("🚀 Enhanced Chat Endpoints Test Server")
    print("Available endpoints:")
    print("  POST /api/chat/test - Test endpoint")
    print("  POST /api/chat/enhanced - Enhanced chat")
    print("  POST /api/chat/simple - Simple chat")
    print("  GET  /api/chat/test - Status check")
    
    enhanced_app.run(host='0.0.0.0', port=5001, debug=True)