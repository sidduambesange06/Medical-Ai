#!/usr/bin/env python3
"""
Enhanced Chat Endpoints with Fixed Error Handling
=================================================
FastAPI router providing /api/chat/test, /api/chat/enhanced, and /api/chat/simple.
Registered on the main FastAPI app in app_production.py via:
    from enhanced_chat_endpoints import create_enhanced_chat_router
    app.include_router(create_enhanced_chat_router(...))
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

# Add project root to path so sibling modules resolve correctly
sys.path.insert(0, os.getcwd())


def create_enhanced_chat_router(
    medical_chatbot=None,
    ai_system=None,
    user_manager=None,
) -> APIRouter:
    """Return a FastAPI APIRouter with enhanced chat endpoints."""

    router = APIRouter(prefix="/api/chat", tags=["chat"])

    @router.get("/test")
    @router.post("/test")
    async def test_chat_endpoint(request: Request):
        """Test endpoint to verify chat functionality."""
        try:
            if request.method == "GET":
                return JSONResponse({
                    "status": "Chat endpoint is working",
                    "medical_chatbot_available": medical_chatbot is not None,
                    "ai_system_available": ai_system is not None,
                    "timestamp": datetime.now().isoformat(),
                })

            try:
                data = await request.json()
            except Exception:
                data = {}
            message = data.get("message", "Test message")

            return JSONResponse({
                "response": f"Echo: {message}",
                "timestamp": datetime.now().isoformat(),
                "method": "test_endpoint",
            })

        except Exception as e:
            return JSONResponse(
                {"error": f"Test endpoint error: {str(e)}", "timestamp": datetime.now().isoformat()},
                status_code=500,
            )

    @router.post("/enhanced")
    async def enhanced_chat_endpoint(request: Request):
        """Enhanced chat endpoint with comprehensive error handling."""
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    data = await request.json()
                except Exception:
                    return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
                message = data.get("message", "").strip()
            else:
                form = await request.form()
                message = form.get("message", "").strip()

            if not message:
                return JSONResponse({"error": "Message is required"}, status_code=400)

            # Read session from cookies (FastAPI cookie-based sessions)
            user_email = request.cookies.get("user_email", "anonymous")
            user_id = request.cookies.get("user_id", "anonymous")
            session_id = request.cookies.get(
                "session_id",
                f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

            user_context = {
                "id": user_id,
                "email": user_email,
                "role": "authenticated" if user_email != "anonymous" else "guest",
            }

            ai_response = "I'm MedAI, ready to help with your medical questions."
            processing_method = "fallback"

            if medical_chatbot:
                try:
                    ai_response = await medical_chatbot.process_query_with_context(
                        message, user_context, session_id
                    )
                    processing_method = "medical_chatbot"
                except Exception as chatbot_error:
                    logging.error(f"Medical chatbot error: {chatbot_error}")
                    if ai_system:
                        try:
                            ai_response = ai_system.process_intelligent_query_with_patient_context(
                                message, user_id, "", user_context
                            )
                            processing_method = "ai_system_fallback"
                        except Exception as ai_error:
                            logging.error(f"AI system fallback error: {ai_error}")
                            ai_response = (
                                f"I apologize, but I'm experiencing technical difficulties. "
                                f"Your message has been received. Please try again."
                            )
                            processing_method = "error_fallback"

            if user_email != "anonymous" and user_manager:
                try:
                    user_manager.save_chat_message(user_email, message, ai_response, session_id)
                except Exception as save_error:
                    logging.warning(f"Chat save failed: {save_error}")

            return JSONResponse({
                "response": ai_response,
                "message_id": f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_type": "authenticated" if user_email != "anonymous" else "guest",
                "processing_method": processing_method,
                "medical_response": True,
            })

        except Exception as e:
            logging.error(f"Enhanced chat endpoint error: {e}")
            return JSONResponse(
                {"error": f"Chat processing failed: {str(e)}", "timestamp": datetime.now().isoformat()},
                status_code=500,
            )

    @router.post("/simple")
    async def simple_chat_endpoint(request: Request):
        """Simple chat endpoint for testing."""
        try:
            content_type = request.headers.get("content-type", "")
            if "application/json" in content_type:
                try:
                    data = await request.json()
                except Exception:
                    data = {}
                message = data.get("message", "")
            else:
                form = await request.form()
                message = form.get("message", "")

            if not message:
                return JSONResponse({"error": "Message required"}, status_code=400)

            return JSONResponse({
                "response": f"I received your message: '{message}'. This is a simple test response from MedAI.",
                "timestamp": datetime.now().isoformat(),
                "endpoint": "simple",
            })

        except Exception as e:
            return JSONResponse(
                {"error": f"Simple chat error: {str(e)}", "timestamp": datetime.now().isoformat()},
                status_code=500,
            )

    return router
