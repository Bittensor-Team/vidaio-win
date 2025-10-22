#!/usr/bin/env python3
"""
Torchvision compatibility patch for Real-ESRGAN
Fixes the missing functional_tensor module in newer torchvision versions
"""

import sys
import types

def patch_torchvision():
    """Patch torchvision to add missing functional_tensor module"""
    try:
        import torchvision.transforms.functional as F
        
        # Check if functional_tensor already exists
        if hasattr(F, 'functional_tensor'):
            return True
            
        # Create the missing functional_tensor module
        functional_tensor = types.ModuleType('functional_tensor')
        
        # Add the _is_tracing function that Real-ESRGAN expects
        def _is_tracing():
            return False
            
        functional_tensor._is_tracing = _is_tracing
        
        # Add other functions that might be needed
        functional_tensor._get_image_size = lambda img: img.size if hasattr(img, 'size') else (img.shape[-1], img.shape[-2])
        functional_tensor._get_image_num_channels = lambda img: img.shape[-3] if len(img.shape) > 2 else 1
        
        # Attach it to the functional module
        F.functional_tensor = functional_tensor
        
        print("✅ Torchvision compatibility patch applied")
        return True
        
    except ImportError as e:
        print(f"❌ Could not patch torchvision: {e}")
        return False

def patch_basicsr():
    """Patch BasicSR to work with newer torchvision"""
    try:
        import basicsr.archs.rrdbnet_arch as rrdbnet_arch
        
        # Check if the module needs patching
        if hasattr(rrdbnet_arch, 'functional_tensor'):
            return True
            
        # Import the patched functional_tensor
        patch_torchvision()
        import torchvision.transforms.functional as F
        
        # Add it to the rrdbnet_arch module
        rrdbnet_arch.functional_tensor = F.functional_tensor
        
        print("✅ BasicSR compatibility patch applied")
        return True
        
    except ImportError as e:
        print(f"❌ Could not patch BasicSR: {e}")
        return False

def apply_all_patches():
    """Apply all compatibility patches"""
    print("🔧 Applying compatibility patches...")
    
    success1 = patch_torchvision()
    success2 = patch_basicsr()
    
    if success1 and success2:
        print("✅ All patches applied successfully")
        return True
    else:
        print("⚠️  Some patches failed, but continuing...")
        return False

if __name__ == "__main__":
    apply_all_patches()
