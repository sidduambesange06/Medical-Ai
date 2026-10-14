#!/usr/bin/env python3
"""
EMERGENCY FAST STARTUP CONFIGURATION
====================================
Optimizes MedBot for instant startup by disabling heavy components
"""

import os
import logging

# ==================== EMERGENCY PERFORMANCE MODE ====================
# Disable heavy components for instant startup

# 1. Disable heavy AI model loading
os.environ['DISABLE_SENTENCE_TRANSFORMERS'] = 'true'
os.environ['DISABLE_PINECONE'] = 'true' 
os.environ['DISABLE_FAISS'] = 'true'
os.environ['LAZY_LOAD_AI'] = 'true'

# 2. Minimal logging
os.environ['MINIMAL_LOGGING'] = 'true'
os.environ['DISABLE_FILE_LOGGING'] = 'true'

# 3. Fast startup mode
os.environ['FAST_STARTUP_MODE'] = 'true'
os.environ['SKIP_HEAVY_INIT'] = 'true'

# 4. Cache optimizations
os.environ['USE_MEMORY_CACHE'] = 'true'
os.environ['DISABLE_REDIS_INIT'] = 'false'  # Keep Redis but don't wait

# 5. Development mode settings
os.environ['DEVELOPMENT_MODE'] = 'true'
os.environ['DEBUG_ROUTES'] = 'true'

print("⚡ EMERGENCY FAST STARTUP MODE ACTIVATED")
print("🚀 Expected startup time: <10 seconds")
print("📝 Heavy AI components will load lazily on first use")