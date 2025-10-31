#!/usr/bin/env python3
"""
Test R2 Upload
Test uploading the processed video to R2
"""

import os
from minio import Minio
from datetime import datetime, timedelta

# R2 credentials
R2_ACCESS_KEY = "eacd80b3fcf1a6d0c98572134a05680c"
R2_SECRET_KEY = "b5a4e900e924f029dd804f096a765c9cc590d997bb25aa7dd0e0fa78ca36ed5f"
R2_ENDPOINT = "ed23ab68357ef85e85a67a8fa27fab47.r2.cloudflarestorage.com"
R2_BUCKET = "vidaio"

def test_r2_upload():
    """Test uploading a file to R2"""
    
    # Initialize R2 client
    client = Minio(R2_ENDPOINT, 
                   access_key=R2_ACCESS_KEY, 
                   secret_key=R2_SECRET_KEY)
    
    # Find the latest processed video
    test_output_dir = "/workspace/vidaio-win/test_output"
    video_files = [f for f in os.listdir(test_output_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print("❌ No video files found in test_output")
        return
    
    # Get the latest file
    latest_file = max(video_files, key=lambda f: os.path.getmtime(os.path.join(test_output_dir, f)))
    local_path = os.path.join(test_output_dir, latest_file)
    
    print(f"📹 Uploading: {latest_file}")
    print(f"📊 File size: {os.path.getsize(local_path) / (1024*1024):.2f} MB")
    
    try:
        # Upload to R2
        client.fput_object(R2_BUCKET, latest_file, local_path)
        print(f"✅ Upload successful: {latest_file}")
        
        # Generate presigned URL
        presigned_url = client.presigned_get_object(R2_BUCKET, latest_file, expires=timedelta(days=7))
        print(f"🔗 Presigned URL: {presigned_url}")
        
        # List objects to verify
        objects = list(client.list_objects(R2_BUCKET, prefix=latest_file))
        if objects:
            obj = objects[0]
            print(f"✅ Verified in R2: {obj.object_name} - {obj.size} bytes")
        else:
            print("❌ File not found in R2 after upload")
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    test_r2_upload()



