"""
Cost Monitor for Bharat-FM Inference Optimization
Tracks and optimizes inference costs across different models and strategies
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class CostMetrics:
    """Cost metrics for inference operations"""
    total_cost: float = 0.0
    total_tokens: int = 0
    total_requests: int = 0
    avg_cost_per_request: float = 0.0
    avg_cost_per_token: float = 0.0
    peak_hourly_cost: float = 0.0
    
@dataclass
class ModelCostProfile:
    """Cost profile for a specific model"""
    model_id: str
    input_token_cost: float  # Cost per 1K input tokens
    output_token_cost: float  # Cost per 1K output tokens
    base_cost_per_request: float  # Fixed cost per request
    compute_cost_per_second: float  # Cost per second of compute
    memory_cost_per_gb: float  # Cost per GB of memory usage

class CostMonitor:
    """Cost monitoring and optimization system"""
    
    def __init__(self):
        # Cost tracking
        self.cost_metrics = defaultdict(CostMetrics)
        self.hourly_costs = defaultdict(lambda: defaultdict(float))
        self.model_costs = defaultdict(lambda: defaultdict(float))
        
        # Model cost profiles (default values)
        self.model_profiles = {
            "default": ModelCostProfile(
                model_id="default",
                input_token_cost=0.001,  # $0.001 per 1K input tokens
                output_token_cost=0.002,  # $0.002 per 1K output tokens
                base_cost_per_request=0.0001,  # $0.0001 per request
                compute_cost_per_second=0.0001,  # $0.0001 per second
                memory_cost_per_gb=0.00001  # $0.00001 per GB per second
            )
        }
        
        # Cost optimization
        self.cost_thresholds = {
            "daily_limit": 100.0,  # $100 daily limit
            "hourly_limit": 10.0,  # $10 hourly limit
            "request_cost_limit": 0.01  # $0.01 per request limit
        }
        
        # Alerting
        self.cost_alerts = deque(maxlen=100)
        self.alert_callbacks = []
        
        # Background tasks
        self.monitor_task = None
        self.running = False
        
    async def start(self):
        """Start cost monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_costs())
        
        logger.info("Cost monitor started")
    
    async def stop(self):
        """Stop cost monitoring"""
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cost monitor stopped")
    
    async def calculate_cost(self, model_info: Dict[str, Any], response: Dict[str, Any]) -> float:
        """Calculate cost for inference request"""
        model_id = model_info.get("id", "default")
        profile = self.model_profiles.get(model_id, self.model_profiles["default"])
        
        # Extract usage metrics
        input_tokens = response.get("usage", {}).get("input_tokens", 0)
        output_tokens = response.get("usage", {}).get("output_tokens", 0)
        compute_time = response.get("compute_time", 0.0)
        memory_usage_gb = response.get("memory_usage_gb", 0.0)
        
        # Calculate component costs
        input_cost = (input_tokens / 1000) * profile.input_token_cost
        output_cost = (output_tokens / 1000) * profile.output_token_cost
        base_cost = profile.base_cost_per_request
        compute_cost = compute_time * profile.compute_cost_per_second
        memory_cost = memory_usage_gb * compute_time * profile.memory_cost_per_gb
        
        total_cost = input_cost + output_cost + base_cost + compute_cost + memory_cost
        
        # Record cost
        await self._record_cost(model_id, total_cost, input_tokens + output_tokens, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "compute_time": compute_time,
            "memory_usage_gb": memory_usage_gb
        })
        
        return total_cost
    
    async def _record_cost(self, model_id: str, cost: float, tokens: int, breakdown: Dict[str, Any]):
        """Record cost data"""
        now = datetime.utcnow()
        hour_key = now.strftime("%Y-%m-%d-%H")
        
        # Update cost metrics
        metrics = self.cost_metrics[model_id]
        metrics.total_cost += cost
        metrics.total_tokens += tokens
        metrics.total_requests += 1
        metrics.avg_cost_per_request = metrics.total_cost / metrics.total_requests
        metrics.avg_cost_per_token = metrics.total_cost / max(1, metrics.total_tokens)
        
        # Update hourly costs
        self.hourly_costs[hour_key][model_id] += cost
        
        # Update model costs
        self.model_costs[model_id][hour_key] += cost
        
        # Check thresholds
        await self._check_cost_thresholds(model_id, cost)
    
    async def _check_cost_thresholds(self, model_id: str, cost: float):
        """Check if cost thresholds are exceeded"""
        metrics = self.cost_metrics[model_id]
        
        # Check per-request cost limit
        if cost > self.cost_thresholds["request_cost_limit"]:
            alert = {
                "type": "high_request_cost",
                "model_id": model_id,
                "cost": cost,
                "threshold": self.cost_thresholds["request_cost_limit"],
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._trigger_alert(alert)
        
        # Check hourly cost limit
        current_hour = datetime.utcnow().strftime("%Y-%m-%d-%H")
        hourly_cost = self.hourly_costs[current_hour].get(model_id, 0)
        if hourly_cost > self.cost_thresholds["hourly_limit"]:
            alert = {
                "type": "high_hourly_cost",
                "model_id": model_id,
                "hourly_cost": hourly_cost,
                "threshold": self.cost_thresholds["hourly_limit"],
                "timestamp": datetime.utcnow().isoformat()
            }
            await self._trigger_alert(alert)
    
    async def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger cost alert"""
        self.cost_alerts.append(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in cost alert callback: {e}")
        
        logger.warning(f"Cost alert triggered: {alert}")
    
    def add_alert_callback(self, callback):
        """Add callback for cost alerts"""
        self.alert_callbacks.append(callback)
    
    async def get_cost_report(self, model_id: Optional[str] = None, 
                            hours: int = 24) -> Dict[str, Any]:
        """Generate cost report"""
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)
        
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": now.isoformat(),
                "hours": hours
            },
            "summary": {},
            "model_breakdown": {},
            "hourly_breakdown": {},
            "trends": {}
        }
        
        # Calculate summary
        total_cost = 0
        total_tokens = 0
        total_requests = 0
        
        for mid, metrics in self.cost_metrics.items():
            if model_id and mid != model_id:
                continue
            
            total_cost += metrics.total_cost
            total_tokens += metrics.total_tokens
            total_requests += metrics.total_requests
        
        report["summary"] = {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "avg_cost_per_request": total_cost / max(1, total_requests),
            "avg_cost_per_token": total_cost / max(1, total_tokens)
        }
        
        # Model breakdown
        for mid, metrics in self.cost_metrics.items():
            if model_id and mid != model_id:
                continue
            
            report["model_breakdown"][mid] = {
                "total_cost": metrics.total_cost,
                "total_tokens": metrics.total_tokens,
                "total_requests": metrics.total_requests,
                "avg_cost_per_request": metrics.avg_cost_per_request,
                "avg_cost_per_token": metrics.avg_cost_per_token,
                "percentage_of_total": (metrics.total_cost / max(1, total_cost)) * 100
            }
        
        # Hourly breakdown
        for hour in range(hours):
            hour_time = now - timedelta(hours=hour)
            hour_key = hour_time.strftime("%Y-%m-%d-%H")
            
            hourly_total = sum(self.hourly_costs[hour_key].values())
            report["hourly_breakdown"][hour_key] = {
                "total_cost": hourly_total,
                "model_costs": dict(self.hourly_costs[hour_key])
            }
        
        # Calculate trends
        report["trends"] = await self._calculate_cost_trends(hours)
        
        return report
    
    async def _calculate_cost_trends(self, hours: int) -> Dict[str, Any]:
        """Calculate cost trends"""
        now = datetime.utcnow()
        
        # Calculate moving averages
        recent_costs = []
        for hour in range(min(24, hours)):
            hour_time = now - timedelta(hours=hour)
            hour_key = hour_time.strftime("%Y-%m-%d-%H")
            hourly_total = sum(self.hourly_costs[hour_key].values())
            recent_costs.append(hourly_total)
        
        if len(recent_costs) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple trend calculation
        recent_avg = sum(recent_costs[-6:]) / min(6, len(recent_costs))  # Last 6 hours
        earlier_avg = sum(recent_costs[-12:-6]) / max(1, len(recent_costs[-12:-6]))  # Previous 6 hours
        
        trend_direction = "increasing" if recent_avg > earlier_avg else "decreasing"
        trend_magnitude = abs(recent_avg - earlier_avg) / max(1, earlier_avg) * 100
        
        return {
            "direction": trend_direction,
            "magnitude_percent": trend_magnitude,
            "recent_avg_hourly": recent_avg,
            "earlier_avg_hourly": earlier_avg
        }
    
    async def get_cost_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Generate cost optimization suggestions"""
        suggestions = []
        
        # Analyze model costs
        for model_id, metrics in self.cost_metrics.items():
            # High cost per request
            if metrics.avg_cost_per_request > 0.005:  # $0.005 per request
                suggestions.append({
                    "type": "high_request_cost",
                    "model_id": model_id,
                    "issue": f"High average cost per request: ${metrics.avg_cost_per_request:.4f}",
                    "suggestion": "Consider batching requests or using a more efficient model",
                    "potential_savings": "20-40%"
                })
            
            # High cost per token
            if metrics.avg_cost_per_token > 0.0001:  # $0.0001 per token
                suggestions.append({
                    "type": "high_token_cost",
                    "model_id": model_id,
                    "issue": f"High average cost per token: ${metrics.avg_cost_per_token:.6f}",
                    "suggestion": "Consider using a smaller model or optimizing prompt length",
                    "potential_savings": "15-30%"
                })
        
        # Analyze usage patterns
        peak_hours = self._identify_peak_hours()
        if peak_hours:
            suggestions.append({
                "type": "peak_usage",
                "issue": f"Peak usage during hours: {', '.join(peak_hours)}",
                "suggestion": "Consider pre-loading models during peak hours or using reserved instances",
                "potential_savings": "10-25%"
            })
        
        return suggestions
    
    def _identify_peak_hours(self) -> List[str]:
        """Identify peak usage hours"""
        hourly_totals = defaultdict(float)
        
        for hour_key, costs in self.hourly_costs.items():
            hour = hour_key.split("-")[3]  # Extract hour
            hourly_totals[hour] += sum(costs.values())
        
        if not hourly_totals:
            return []
        
        # Find hours with cost above 75th percentile
        costs = list(hourly_totals.values())
        threshold = sorted(costs)[int(len(costs) * 0.75)] if costs else 0
        
        peak_hours = [hour for hour, total in hourly_totals.items() if total > threshold]
        return peak_hours
    
    async def _monitor_costs(self):
        """Background task for continuous cost monitoring"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Generate optimization suggestions
                suggestions = await self.get_cost_optimization_suggestions()
                
                # Log significant findings
                for suggestion in suggestions:
                    if suggestion["type"] in ["high_request_cost", "high_token_cost"]:
                        logger.info(f"Cost optimization suggestion: {suggestion}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cost monitoring: {e}")
    
    async def get_patterns(self) -> Dict[str, Any]:
        """Get cost patterns and insights"""
        return {
            "hourly_patterns": dict(self.hourly_costs),
            "model_patterns": dict(self.model_costs),
            "cost_metrics": {model_id: dict(metrics) for model_id, metrics in self.cost_metrics.items()},
            "recent_alerts": list(self.cost_alerts)[-10:],  # Last 10 alerts
            "optimization_suggestions": await self.get_cost_optimization_suggestions()
        }