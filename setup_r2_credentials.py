#!/usr/bin/env python3
"""
Setup R2 S3-Compatible Credentials
Creates S3-compatible credentials for Cloudflare R2 using API tokens
"""

import requests
import json
import os

# Cloudflare API configuration
ACCOUNT_ID = "ed23ab68357ef85e85a67a8fa27fab47"
API_TOKEN = "oyo1cZKLytzmYjY0N0EFCVaVw3O88E0RSsyd96vK"
BUCKET_NAME = "vidaio"

def get_r2_s3_credentials():
    """Get S3-compatible credentials for R2"""
    
    # First, let's try to list R2 buckets to see what's available
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/r2/buckets"
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("🔍 Checking R2 buckets...")
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ R2 API access successful")
        print(f"📊 Response: {json.dumps(data, indent=2)}")
        
        # Check if our bucket exists
        if 'result' in data and isinstance(data['result'], list):
            bucket_names = [bucket['name'] for bucket in data['result']]
            print(f"📦 Available buckets: {bucket_names}")
            
            if BUCKET_NAME in bucket_names:
                print(f"✅ Bucket '{BUCKET_NAME}' found")
            else:
                print(f"❌ Bucket '{BUCKET_NAME}' not found")
                print("Available buckets:", bucket_names)
        else:
            print(f"⚠️  Unexpected response format: {data}")
            
    else:
        print(f"❌ Failed to access R2 API: {response.status_code}")
        print(f"Response: {response.text}")
        return None
    
    # For S3-compatible access, we need to create R2 API tokens
    # Let's try to create S3 credentials
    print("\n🔑 Creating S3-compatible credentials...")
    
    # Note: The API tokens you provided are for Cloudflare API access, not S3-compatible access
    # For S3-compatible access, you need to create R2 API tokens in the Cloudflare dashboard
    # that specifically provide S3 credentials (Access Key ID and Secret Access Key)
    
    print("⚠️  Current tokens are Cloudflare API tokens, not S3-compatible credentials")
    print("📋 To get S3-compatible credentials:")
    print("   1. Go to Cloudflare Dashboard > R2 > Manage R2 API Tokens")
    print("   2. Create a new token with 'R2:Edit' permissions")
    print("   3. Select 'S3 API' as the token type")
    print("   4. This will give you Access Key ID and Secret Access Key")
    
    return None

def test_public_url():
    """Test the public development URL"""
    public_url = "https://pub-f528bcc525c94b69a931dd75f8d233cc.r2.dev"
    
    print(f"\n🌐 Testing public URL: {public_url}")
    
    try:
        response = requests.get(public_url, timeout=10)
        print(f"📊 Public URL status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Public URL is accessible")
        else:
            print("⚠️  Public URL returned non-200 status")
    except Exception as e:
        print(f"❌ Public URL test failed: {e}")

if __name__ == "__main__":
    print("🚀 Setting up R2 Credentials")
    print("=" * 50)
    
    get_r2_s3_credentials()
    test_public_url()
    
    print("\n📋 Next Steps:")
    print("1. Create S3-compatible R2 API tokens in Cloudflare dashboard")
    print("2. Set environment variables:")
    print("   export R2_ACCESS_KEY='your_s3_access_key'")
    print("   export R2_SECRET_KEY='your_s3_secret_key'")
    print("3. Test with: python test_r2_access.py")
