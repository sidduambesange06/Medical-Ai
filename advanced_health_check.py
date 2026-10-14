#!/usr/bin/env python3
"""
Advanced Health Check & System Monitoring for MedBot Ultra v3.0
Comprehensive system validation, performance monitoring, and diagnostic utilities
"""

import os
import sys
import json
import time
import psutil
import requests
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class AdvancedHealthChecker:
    """Comprehensive health checking and system validation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.health_status = {
            'overall': 'UNKNOWN',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'metrics': {},
            'alerts': []
        }
    
    def run_comprehensive_check(self) -> Dict:
        """Run all health checks and return comprehensive status"""
        print("🔍 Starting Advanced Health Check for MedBot Ultra v3.0")
        print("=" * 70)
        
        # Core system checks
        self.check_system_resources()
        self.check_python_environment()
        self.check_dependencies()
        self.check_configuration()
        self.check_file_structure()
        self.check_database_connectivity()
        self.check_ai_services()
        self.check_security_status()
        
        # Calculate overall status
        self.calculate_overall_status()
        
        print("\n" + "=" * 70)
        print(f"🎯 Health Check Complete: {self.health_status['overall']}")
        print(f"⏱️  Total Check Time: {time.time() - self.start_time:.2f}s")
        
        return self.health_status
    
    def check_system_resources(self):
        """Check system resource availability"""
        print("\n🖥️  System Resources Check...")
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # CPU check
            cpu_usage = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Disk check
            disk_usage = psutil.disk_usage('/')
            disk_free_gb = disk_usage.free / (1024**3)
            disk_usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            # Network check
            network_stats = psutil.net_io_counters()
            
            status = "HEALTHY"
            alerts = []
            
            if memory_usage > 85:
                status = "WARNING"
                alerts.append(f"High memory usage: {memory_usage}%")
            
            if cpu_usage > 80:
                status = "WARNING"
                alerts.append(f"High CPU usage: {cpu_usage}%")
            
            if disk_usage_percent > 90:
                status = "CRITICAL"
                alerts.append(f"Low disk space: {disk_usage_percent:.1f}% used")
            
            if memory_available_gb < 0.5:
                status = "CRITICAL"
                alerts.append(f"Low memory: {memory_available_gb:.1f}GB available")
            
            self.health_status['checks']['system_resources'] = {
                'status': status,
                'memory': {
                    'usage_percent': memory_usage,
                    'available_gb': round(memory_available_gb, 2),
                    'total_gb': round(memory.total / (1024**3), 2)
                },
                'cpu': {
                    'usage_percent': cpu_usage,
                    'core_count': cpu_count
                },
                'disk': {
                    'usage_percent': round(disk_usage_percent, 1),
                    'free_gb': round(disk_free_gb, 2),
                    'total_gb': round(disk_usage.total / (1024**3), 2)
                },
                'network': {
                    'bytes_sent': network_stats.bytes_sent,
                    'bytes_received': network_stats.bytes_recv
                },
                'alerts': alerts
            }
            
            print(f"   ✅ Memory: {memory_usage}% used ({memory_available_gb:.1f}GB available)")
            print(f"   ✅ CPU: {cpu_usage}% usage ({cpu_count} cores)")
            print(f"   ✅ Disk: {disk_usage_percent:.1f}% used ({disk_free_gb:.1f}GB free)")
            
            for alert in alerts:
                print(f"   ⚠️  {alert}")
                self.health_status['alerts'].append(alert)
            
        except Exception as e:
            self.health_status['checks']['system_resources'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"   ❌ System resource check failed: {e}")
    
    def check_python_environment(self):
        """Check Python environment and version"""
        print("\n🐍 Python Environment Check...")
        
        try:
            python_version = sys.version
            python_executable = sys.executable
            python_path = sys.path
            
            # Check Python version compatibility
            version_info = sys.version_info
            is_compatible = version_info.major == 3 and version_info.minor >= 8
            
            status = "HEALTHY" if is_compatible else "WARNING"
            
            self.health_status['checks']['python_environment'] = {
                'status': status,
                'version': python_version.split()[0],
                'executable': python_executable,
                'compatible': is_compatible,
                'version_info': {
                    'major': version_info.major,
                    'minor': version_info.minor,
                    'micro': version_info.micro
                }
            }
            
            print(f"   ✅ Python Version: {python_version.split()[0]}")
            print(f"   ✅ Executable: {python_executable}")
            
            if not is_compatible:
                alert = f"Python version {version_info.major}.{version_info.minor} may not be fully compatible (recommended: 3.8+)"
                print(f"   ⚠️  {alert}")
                self.health_status['alerts'].append(alert)
            
        except Exception as e:
            self.health_status['checks']['python_environment'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"   ❌ Python environment check failed: {e}")
    
    def check_dependencies(self):
        """Check critical dependencies availability"""
        print("\n📦 Dependencies Check...")
        
        critical_deps = [
            'fastapi', 'uvicorn', 'starlette',
            'requests', 'redis', 'supabase', 'groq', 'transformers',
            'torch', 'langchain', 'psutil', 'dotenv'
        ]
        
        optional_deps = [
            'nltk', 'spacy', 'pinecone', 'gunicorn', 'flask_cors'
        ]
        
        missing_critical = []
        missing_optional = []
        available_deps = {}
        
        # Check critical dependencies
        for dep in critical_deps:
            try:
                if dep == 'dotenv':
                    import dotenv
                    module = dotenv
                else:
                    module = __import__(dep)
                
                version = getattr(module, '__version__', 'unknown')
                available_deps[dep] = version
                print(f"   ✅ {dep}: {version}")
                
            except ImportError:
                missing_critical.append(dep)
                print(f"   ❌ {dep}: NOT FOUND")
        
        # Check optional dependencies
        for dep in optional_deps:
            try:
                module = __import__(dep)
                version = getattr(module, '__version__', 'unknown')
                available_deps[dep] = version
                print(f"   🔸 {dep}: {version} (optional)")
                
            except ImportError:
                missing_optional.append(dep)
                print(f"   ⚠️  {dep}: NOT FOUND (optional)")
        
        status = "HEALTHY"
        if missing_critical:
            status = "CRITICAL"
            alert = f"Missing critical dependencies: {', '.join(missing_critical)}"
            self.health_status['alerts'].append(alert)
        elif missing_optional:
            status = "WARNING"
        
        self.health_status['checks']['dependencies'] = {
            'status': status,
            'available': available_deps,
            'missing_critical': missing_critical,
            'missing_optional': missing_optional
        }
    
    def check_configuration(self):
        """Check configuration files and environment variables"""
        print("\n⚙️  Configuration Check...")
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            required_env_vars = [
                'SUPABASE_URL', 'SUPABASE_KEY', 'GROQ_API_KEY',
                'PINECONE_API_KEY', 'SECRET_KEY'
            ]
            
            missing_vars = []
            env_status = {}
            
            for var in required_env_vars:
                value = os.getenv(var)
                if value:
                    # Don't log actual values for security
                    env_status[var] = "✓ SET"
                    print(f"   ✅ {var}: SET")
                else:
                    missing_vars.append(var)
                    env_status[var] = "✗ MISSING"
                    print(f"   ❌ {var}: MISSING")
            
            # Check config files
            config_files = ['.env', 'app.py']
            file_status = {}
            
            for file in config_files:
                if Path(file).exists():
                    file_status[file] = "✓ EXISTS"
                    print(f"   ✅ {file}: EXISTS")
                else:
                    file_status[file] = "✗ MISSING"
                    print(f"   ❌ {file}: MISSING")
            
            status = "HEALTHY"
            if missing_vars:
                status = "CRITICAL"
                alert = f"Missing environment variables: {', '.join(missing_vars)}"
                self.health_status['alerts'].append(alert)
            
            self.health_status['checks']['configuration'] = {
                'status': status,
                'environment_variables': env_status,
                'config_files': file_status,
                'missing_vars': missing_vars
            }
            
        except Exception as e:
            self.health_status['checks']['configuration'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"   ❌ Configuration check failed: {e}")
    
    def check_file_structure(self):
        """Check essential file structure"""
        print("\n📁 File Structure Check...")
        
        required_files = [
            'app.py', 'requirements_production.txt',
            'templates/chat_new_premium.html',
            'templates/admin_dashboard.html',
            'static/js/superbase.js'
        ]
        
        required_dirs = [
            'templates', 'static', 'data', 'logs'
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file in required_files:
            if Path(file).exists():
                print(f"   ✅ {file}: EXISTS")
            else:
                missing_files.append(file)
                print(f"   ❌ {file}: MISSING")
        
        for dir in required_dirs:
            if Path(dir).exists():
                print(f"   ✅ {dir}/: EXISTS")
            else:
                missing_dirs.append(dir)
                print(f"   ❌ {dir}/: MISSING")
        
        status = "HEALTHY"
        if missing_files or missing_dirs:
            status = "WARNING"
            if 'app.py' in missing_files:
                status = "CRITICAL"
        
        self.health_status['checks']['file_structure'] = {
            'status': status,
            'missing_files': missing_files,
            'missing_dirs': missing_dirs
        }
    
    def check_database_connectivity(self):
        """Check database and external service connectivity"""
        print("\n🗄️  Database Connectivity Check...")
        
        services = {
            'redis': self._check_redis(),
            'supabase': self._check_supabase()
        }
        
        all_healthy = all(service['status'] == 'HEALTHY' for service in services.values())
        status = "HEALTHY" if all_healthy else "WARNING"
        
        self.health_status['checks']['database_connectivity'] = {
            'status': status,
            'services': services
        }
    
    def _check_redis(self) -> Dict:
        """Check Redis connectivity"""
        try:
            import redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            r = redis.from_url(redis_url)
            r.ping()
            print("   ✅ Redis: CONNECTED")
            return {'status': 'HEALTHY', 'url': redis_url}
        except Exception as e:
            print(f"   ⚠️  Redis: UNAVAILABLE ({e})")
            return {'status': 'WARNING', 'error': str(e)}
    
    def _check_supabase(self) -> Dict:
        """Check Supabase connectivity"""
        try:
            supabase_url = os.getenv('SUPABASE_URL')
            if not supabase_url:
                print("   ❌ Supabase: NO URL CONFIGURED")
                return {'status': 'ERROR', 'error': 'No URL configured'}
            
            # Simple health check request
            response = requests.get(f"{supabase_url}/rest/v1/", timeout=5)
            if response.status_code < 500:
                print("   ✅ Supabase: CONNECTED")
                return {'status': 'HEALTHY', 'url': supabase_url}
            else:
                print(f"   ⚠️  Supabase: ISSUES (status: {response.status_code})")
                return {'status': 'WARNING', 'status_code': response.status_code}
                
        except Exception as e:
            print(f"   ⚠️  Supabase: UNAVAILABLE ({e})")
            return {'status': 'WARNING', 'error': str(e)}
    
    def check_ai_services(self):
        """Check AI service availability"""
        print("\n🤖 AI Services Check...")
        
        services = {
            'openai': self._check_openai(),
            'groq': self._check_groq()
        }
        
        healthy_services = [name for name, service in services.items() if service['status'] == 'HEALTHY']
        status = "HEALTHY" if healthy_services else "WARNING"
        
        self.health_status['checks']['ai_services'] = {
            'status': status,
            'services': services,
            'healthy_count': len(healthy_services)
        }
    
    def _check_openai(self) -> Dict:
        """Check OpenAI API availability"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("   ⚠️  OpenAI: NO API KEY")
                return {'status': 'WARNING', 'error': 'No API key'}
            
            print("   ✅ OpenAI: API KEY CONFIGURED")
            return {'status': 'HEALTHY'}
            
        except Exception as e:
            print(f"   ❌ OpenAI: ERROR ({e})")
            return {'status': 'ERROR', 'error': str(e)}
    
    def _check_groq(self) -> Dict:
        """Check Groq API availability"""
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                print("   ❌ Groq: NO API KEY")
                return {'status': 'ERROR', 'error': 'No API key'}
            
            print("   ✅ Groq: API KEY CONFIGURED")
            return {'status': 'HEALTHY'}
            
        except Exception as e:
            print(f"   ❌ Groq: ERROR ({e})")
            return {'status': 'ERROR', 'error': str(e)}
    
    def check_security_status(self):
        """Check security configuration"""
        print("\n🛡️  Security Status Check...")
        
        try:
            security_checks = {
                'secret_key': bool(os.getenv('SECRET_KEY')),
                'jwt_secret': bool(os.getenv('JWT_SECRET')),
                'https_config': os.getenv('FLASK_ENV') == 'production',
                'debug_disabled': os.getenv('FLASK_DEBUG', 'false').lower() != 'true'
            }
            
            security_score = sum(security_checks.values())
            total_checks = len(security_checks)
            
            if security_score == total_checks:
                status = "HEALTHY"
                print("   ✅ All security checks passed")
            elif security_score >= total_checks * 0.75:
                status = "WARNING"
                print("   ⚠️  Some security issues found")
            else:
                status = "CRITICAL"
                print("   ❌ Critical security issues found")
            
            for check, passed in security_checks.items():
                symbol = "✅" if passed else "❌"
                print(f"   {symbol} {check}: {'PASS' if passed else 'FAIL'}")
            
            self.health_status['checks']['security'] = {
                'status': status,
                'checks': security_checks,
                'score': f"{security_score}/{total_checks}"
            }
            
        except Exception as e:
            self.health_status['checks']['security'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"   ❌ Security check failed: {e}")
    
    def calculate_overall_status(self):
        """Calculate overall health status based on all checks"""
        statuses = [check.get('status', 'UNKNOWN') for check in self.health_status['checks'].values()]
        
        if 'CRITICAL' in statuses:
            self.health_status['overall'] = 'CRITICAL'
        elif 'ERROR' in statuses:
            self.health_status['overall'] = 'ERROR'
        elif 'WARNING' in statuses:
            self.health_status['overall'] = 'WARNING'
        elif all(status == 'HEALTHY' for status in statuses):
            self.health_status['overall'] = 'HEALTHY'
        else:
            self.health_status['overall'] = 'UNKNOWN'
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate detailed health report"""
        report = {
            **self.health_status,
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator': 'MedBot Ultra v3.0 Health Checker',
                'system_info': {
                    'platform': sys.platform,
                    'python_version': sys.version.split()[0],
                    'hostname': os.uname().nodename if hasattr(os, 'uname') else 'unknown'
                }
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Detailed report saved to: {output_file}")
        
        return json.dumps(report, indent=2)


def main():
    """Run the advanced health check"""
    try:
        checker = AdvancedHealthChecker()
        status = checker.run_comprehensive_check()
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"health_check_report_{timestamp}.json"
        checker.generate_report(report_file)
        
        # Exit with appropriate code
        exit_codes = {
            'HEALTHY': 0,
            'WARNING': 1,
            'ERROR': 2,
            'CRITICAL': 3,
            'UNKNOWN': 4
        }
        
        exit_code = exit_codes.get(status['overall'], 4)
        print(f"\n🎯 Exiting with code {exit_code} ({status['overall']})")
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n💥 Health check failed with exception: {e}")
        print(f"📍 Traceback: {traceback.format_exc()}")
        sys.exit(5)


if __name__ == "__main__":
    main()