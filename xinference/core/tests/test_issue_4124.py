"""
Test cases for Issue #4124 fix
Tests the get_devices_count method in distributed deployment scenarios
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from xinference.core.supervisor import SupervisorActor, WorkerStatus
from xinference.core.resource import ResourceStatus, GPUStatus


class TestGetDevicesCount:
    """Test cases for SupervisorActor.get_devices_count()"""
    
    @pytest.mark.asyncio
    async def test_local_deployment_with_gpu(self):
        """Test local deployment when supervisor has GPU"""
        supervisor = SupervisorActor()
        supervisor.address = "localhost:9997"
        supervisor._worker_address_to_worker = {"localhost:9997": MagicMock()}
        
        with patch('xinference.device_utils.gpu_count', return_value=2):
            count = await supervisor.get_devices_count()
            assert count == 2
    
    @pytest.mark.asyncio
    async def test_local_deployment_no_gpu(self):
        """Test local deployment when supervisor has no GPU"""
        supervisor = SupervisorActor()
        supervisor.address = "localhost:9997"
        supervisor._worker_address_to_worker = {"localhost:9997": MagicMock()}
        
        with patch('xinference.device_utils.gpu_count', return_value=0):
            count = await supervisor.get_devices_count()
            assert count == 0
    
    @pytest.mark.asyncio
    async def test_distributed_no_workers(self):
        """Test distributed deployment when no workers have registered"""
        supervisor = SupervisorActor()
        supervisor.address = "192.168.1.1:9997"
        supervisor._worker_address_to_worker = {}
        supervisor._worker_status = {}
        
        with patch('xinference.device_utils.gpu_count', return_value=0):
            count = await supervisor.get_devices_count()
            # Should fallback to local detection
            assert count == 0
    
    @pytest.mark.asyncio
    async def test_distributed_supervisor_no_gpu_worker_has_gpu(self):
        """
        Test Issue #4124 scenario:
        Supervisor has no GPU, but worker has GPUs
        """
        supervisor = SupervisorActor()
        supervisor.address = "192.168.1.33:9997"
        supervisor._worker_address_to_worker = {
            "192.168.1.100:30001": MagicMock()
        }
        
        # Simulate worker status with 2 GPUs
        supervisor._worker_status = {
            "192.168.1.100:30001": WorkerStatus(
                update_time=1234567890.0,
                failure_remaining_count=3,
                status={
                    "cpu": ResourceStatus(
                        usage=0.5,
                        total=8.0,
                        memory_used=8589934592,
                        memory_available=8589934592,
                        memory_total=17179869184
                    ),
                    "gpu-0": GPUStatus(
                        name="NVIDIA GeForce RTX 3090",
                        mem_total=25769803776,
                        mem_free=20615843020,
                        mem_used=5153960756,
                        mem_usage=0.2,
                        gpu_util=10.0
                    ),
                    "gpu-1": GPUStatus(
                        name="NVIDIA GeForce RTX 3090",
                        mem_total=25769803776,
                        mem_free=25769803776,
                        mem_used=0,
                        mem_usage=0.0,
                        gpu_util=0.0
                    )
                }
            )
        }
        
        with patch('xinference.device_utils.gpu_count', return_value=0):
            count = await supervisor.get_devices_count()
            # Should return worker's GPU count (2), not supervisor's (0)
            assert count == 2
    
    @pytest.mark.asyncio
    async def test_distributed_multiple_workers_different_gpu_counts(self):
        """
        Test with multiple workers having different GPU counts
        Should return the maximum
        """
        supervisor = SupervisorActor()
        supervisor.address = "192.168.1.33:9997"
        supervisor._worker_address_to_worker = {
            "192.168.1.100:30001": MagicMock(),
            "192.168.1.101:30001": MagicMock()
        }
        
        # Worker 1 has 2 GPUs
        supervisor._worker_status = {
            "192.168.1.100:30001": WorkerStatus(
                update_time=1234567890.0,
                failure_remaining_count=3,
                status={
                    "cpu": ResourceStatus(
                        usage=0.5, total=8.0,
                        memory_used=8589934592,
                        memory_available=8589934592,
                        memory_total=17179869184
                    ),
                    "gpu-0": GPUStatus(
                        name="GPU", mem_total=1000, mem_free=900,
                        mem_used=100, mem_usage=0.1, gpu_util=10.0
                    ),
                    "gpu-1": GPUStatus(
                        name="GPU", mem_total=1000, mem_free=900,
                        mem_used=100, mem_usage=0.1, gpu_util=10.0
                    )
                }
            ),
            # Worker 2 has 4 GPUs
            "192.168.1.101:30001": WorkerStatus(
                update_time=1234567890.0,
                failure_remaining_count=3,
                status={
                    "cpu": ResourceStatus(
                        usage=0.5, total=16.0,
                        memory_used=8589934592,
                        memory_available=8589934592,
                        memory_total=17179869184
                    ),
                    "gpu-0": GPUStatus(
                        name="GPU", mem_total=1000, mem_free=900,
                        mem_used=100, mem_usage=0.1, gpu_util=10.0
                    ),
                    "gpu-1": GPUStatus(
                        name="GPU", mem_total=1000, mem_free=900,
                        mem_used=100, mem_usage=0.1, gpu_util=10.0
                    ),
                    "gpu-2": GPUStatus(
                        name="GPU", mem_total=1000, mem_free=900,
                        mem_used=100, mem_usage=0.1, gpu_util=10.0
                    ),
                    "gpu-3": GPUStatus(
                        name="GPU", mem_total=1000, mem_free=900,
                        mem_used=100, mem_usage=0.1, gpu_util=10.0
                    )
                }
            )
        }
        
        with patch('xinference.device_utils.gpu_count', return_value=0):
            count = await supervisor.get_devices_count()
            # Should return max GPU count (4)
            assert count == 4
    
    @pytest.mark.asyncio
    async def test_distributed_worker_with_only_cpu(self):
        """Test with worker that has no GPU"""
        supervisor = SupervisorActor()
        supervisor.address = "192.168.1.33:9997"
        supervisor._worker_address_to_worker = {
            "192.168.1.100:30001": MagicMock()
        }
        
        # Worker has only CPU, no GPU
        supervisor._worker_status = {
            "192.168.1.100:30001": WorkerStatus(
                update_time=1234567890.0,
                failure_remaining_count=3,
                status={
                    "cpu": ResourceStatus(
                        usage=0.5, total=8.0,
                        memory_used=8589934592,
                        memory_available=8589934592,
                        memory_total=17179869184
                    )
                }
            )
        }
        
        with patch('xinference.device_utils.gpu_count', return_value=0):
            count = await supervisor.get_devices_count()
            assert count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
