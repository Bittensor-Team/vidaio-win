#!/usr/bin/env python3
"""
Test Cloudflare R2 Access
Verifies that we can connect to and upload to the R2 bucket
"""

import os
import tempfile
import asyncio
from minio import Minio
from datetime import timedelta

# R2 Configuration
R2_ENDPOINT = "https://ed23ab68357ef85e85a67a8fa27fab47.r2.cloudflarestorage.com"
R2_BUCKET = "vidaio"
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")

async def test_r2_connection():
    """Test R2 connection and upload"""
    print("🔍 Testing Cloudflare R2 Connection...")
    
    if not R2_ACCESS_KEY or not R2_SECRET_KEY:
        print("❌ R2 credentials not set. Please set R2_ACCESS_KEY and R2_SECRET_KEY")
        return False
    
    try:
        # Initialize R2 client
        client = Minio(
            R2_ENDPOINT.replace("https://", ""),
            access_key=R2_ACCESS_KEY,
            secret_key=R2_SECRET_KEY,
            secure=True,
            region="auto"
        )
        
        print(f"✅ R2 client initialized")
        print(f"   Endpoint: {R2_ENDPOINT}")
        print(f"   Bucket: {R2_BUCKET}")
        
        # Test bucket access
        if client.bucket_exists(R2_BUCKET):
            print(f"✅ Bucket '{R2_BUCKET}' exists and is accessible")
        else:
            print(f"❌ Bucket '{R2_BUCKET}' does not exist or is not accessible")
            return False
        
        # Create a test file
        test_content = b"Test file for R2 upload verification"
        test_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        test_file.write(test_content)
        test_file.close()
        
        # Upload test file
        object_name = f"test_upload_{int(time.time())}.txt"
        result = client.fput_object(R2_BUCKET, object_name, test_file.name)
        print(f"✅ Test file uploaded: {object_name}")
        print(f"   ETag: {result.etag}")
        
        # Generate presigned URL
        presigned_url = client.presigned_get_object(
            R2_BUCKET, 
            object_name, 
            expires=timedelta(hours=1)
        )
        print(f"✅ Presigned URL generated: {presigned_url[:50]}...")
        
        # Test download
        download_file = tempfile.NamedTemporaryFile(delete=False, suffix="_download.txt")
        download_file.close()
        
        client.fget_object(R2_BUCKET, object_name, download_file.name)
        
        with open(download_file.name, 'rb') as f:
            downloaded_content = f.read()
        
        if downloaded_content == test_content:
            print("✅ Download test successful - content matches")
        else:
            print("❌ Download test failed - content mismatch")
            return False
        
        # Cleanup
        client.remove_object(R2_BUCKET, object_name)
        os.unlink(test_file.name)
        os.unlink(download_file.name)
        print("✅ Test file cleaned up")
        
        print("\n🎉 R2 Access Test PASSED!")
        print("   ✅ Connection successful")
        print("   ✅ Bucket accessible")
        print("   ✅ Upload successful")
        print("   ✅ Download successful")
        print("   ✅ Presigned URL generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ R2 test failed: {e}")
        return False

if __name__ == "__main__":
    import time
    asyncio.run(test_r2_connection())



