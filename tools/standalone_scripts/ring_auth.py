#!/usr/bin/env python3
"""
Ring API Authentication Script

This standalone script helps users authenticate with Ring's API and 
generate a refresh token that can be used with the Ring Tool.

The token can be saved to .env as RING_API_TOKEN.
"""
import json
import os
import re
import requests
import uuid
import platform
import getpass
import time
from typing import Dict, Any, Optional

# Ensure directory exists
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Constants
AUTH_URL = "https://oauth.ring.com/oauth/token"
TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "data", "tools", "ring_tool", "token.json")

def generate_hardware_id() -> str:
    """
    Generate a unique hardware ID for authentication.
    
    Returns:
        A unique hardware ID string
    """
    # Create a deterministic hardware ID based on machine info
    system_info = platform.uname()
    base = f"{system_info.system}-{system_info.node}-{system_info.machine}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

def authenticate_with_email_password(email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate using email and password.
    
    Args:
        email: Ring account email
        password: Ring account password
        
    Returns:
        Authentication response or 2FA response
        
    Raises:
        Exception: If authentication fails
    """
    hardware_id = generate_hardware_id()
    headers = {
        "hardware_id": hardware_id,
        "2fa-support": "true",
        "User-Agent": "android:com.ringapp",
    }
    
    data = {
        "client_id": "ring_official_android",
        "grant_type": "password",
        "username": email,
        "password": password,
        "scope": "client",
    }
    
    response = requests.post(AUTH_URL, headers=headers, json=data)
    
    if response.status_code == 412:  # 2FA required
        # Extract 2FA information
        response_json = response.json()
        tsv_state = response_json.get("tsv_state", "")
        phone = response_json.get("phone", "")
        
        # Determine 2FA method
        if tsv_state == "totp":
            prompt = "from your authenticator app"
        else:
            prompt = f"sent to {phone} via {tsv_state}"
            
        return {
            "type": "2fa_required",
            "prompt": prompt,
            "tsv_state": tsv_state,
        }
    elif not response.ok:
        error_message = f"Authentication failed: {response.status_code} {response.reason}"
        try:
            error_data = response.json()
            if "error_description" in error_data:
                error_message += f" - {error_data['error_description']}"
        except:
            pass
        raise Exception(error_message)
        
    return response.json()

def authenticate_with_2fa(email: str, password: str, code: str) -> Dict[str, Any]:
    """
    Complete authentication with 2FA code.
    
    Args:
        email: Ring account email
        password: Ring account password
        code: 2FA verification code
        
    Returns:
        Authentication response
        
    Raises:
        Exception: If authentication fails
    """
    hardware_id = generate_hardware_id()
    headers = {
        "hardware_id": hardware_id,
        "2fa-support": "true",
        "2fa-code": code,
        "User-Agent": "android:com.ringapp",
    }
    
    data = {
        "client_id": "ring_official_android",
        "grant_type": "password",
        "username": email,
        "password": password,
        "scope": "client",
    }
    
    response = requests.post(AUTH_URL, headers=headers, json=data)
    
    if not response.ok:
        error_message = f"2FA Authentication failed: {response.status_code} {response.reason}"
        try:
            error_data = response.json()
            if "error_description" in error_data:
                error_message += f" - {error_data['error_description']}"
        except:
            pass
        raise Exception(error_message)
        
    return response.json()

def authenticate_with_code(auth_code: str) -> Dict[str, Any]:
    """
    Exchange authorization code for tokens.
    
    Args:
        auth_code: Authorization code from OAuth flow
        
    Returns:
        Dictionary containing token data
        
    Raises:
        Exception: If authentication fails
    """
    headers = {
        "hardware_id": generate_hardware_id(),
        "User-Agent": "android:com.ringapp",
    }
    
    data = {
        "client_id": "ring_official_android",
        "grant_type": "authorization_code",
        "code": auth_code,
    }
    
    response = requests.post(AUTH_URL, headers=headers, json=data)
    
    if not response.ok:
        raise Exception(f"Authentication failed: {response.status_code} {response.reason}")
        
    return response.json()

def save_token(token_data: Dict[str, Any]) -> None:
    """
    Save token data to file.
    
    Args:
        token_data: Token data to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=2)
    print(f"Token saved to {TOKEN_FILE}")

def email_password_flow() -> Optional[Dict[str, Any]]:
    """
    Run the authentication flow using email and password.
    
    Returns:
        Token data if successful, None otherwise
    """
    print("\n*** Ring API Email/Password Authentication ***\n")
    
    email = input("Enter your Ring account email: ").strip()
    password = getpass.getpass("Enter your Ring account password: ")
    
    try:
        # Try initial login
        print("\nAuthenticating with Ring...")
        auth_result = authenticate_with_email_password(email, password)
        
        # Check if 2FA is required
        if auth_result.get("type") == "2fa_required":
            prompt = auth_result.get("prompt", "")
            print(f"\n2FA verification required. Please enter the code {prompt}.")
            
            # Get 2FA code from user
            code = input("Enter 2FA code: ").strip()
            
            # Complete authentication with 2FA
            print("\nCompleting authentication with 2FA...")
            auth_result = authenticate_with_2fa(email, password, code)
        
        # At this point we should have a token
        refresh_token = auth_result.get("refresh_token")
        if not refresh_token:
            print("\nERROR: No refresh token in response!")
            return None
        
        # Save token to file
        save_token(auth_result)
        
        print("\n=== SUCCESS! ===")
        print(f"\nYour refresh token is: {refresh_token}")
        print("\nTo use this token with the Ring Tool, add it to your .env file as:")
        print(f"RING_API_TOKEN={refresh_token}")
        print("\nThis token will not expire unless you manually revoke it from your Ring account.")
        
        return auth_result
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nAuthentication failed. Please try again.")
        return None

def oauth_flow() -> Optional[Dict[str, Any]]:
    """
    Run the authentication flow using OAuth.
    
    Returns:
        Token data if successful, None otherwise
    """
    print("\n*** Ring API OAuth Authentication ***\n")
    print("This method uses the OAuth flow to generate a refresh token.")
    print("\nFollow these steps:")
    
    # Step 1: Open the authorization URL
    print("\n1. Go to this URL in your web browser:")
    print("https://oauth.ring.com/oauth/authorize?client_id=ring_official_android&response_type=code&scope=client")
    
    print("\n2. Log in with your Ring credentials")
    print("\n3. After logging in, you'll be redirected to a URL that starts with 'com.ringapp://'")
    print("   You may see a message about opening the Ring app - you can ignore this.")
    print("\n4. Copy the ENTIRE URL from your browser's address bar")
    
    # Step 2: Get the code from the URL
    auth_url = input("\nPaste the URL here: ").strip()
    match = re.search(r"code=([^&]+)", auth_url)
    if not match:
        print("\nERROR: Could not find authorization code in the URL.")
        print("Make sure you copied the entire URL.")
        return None
        
    auth_code = match.group(1)
    print(f"\nFound authorization code: {auth_code[:5]}...{auth_code[-5:]}")
    
    # Step 3: Exchange code for token
    try:
        print("\nExchanging code for token...")
        token_data = authenticate_with_code(auth_code)
        refresh_token = token_data["refresh_token"]
        
        # Save token to file
        save_token(token_data)
        
        # Display token for user
        print("\n=== SUCCESS! ===")
        print(f"\nYour refresh token is: {refresh_token}")
        print("\nTo use this token with the Ring Tool, add it to your .env file as:")
        print(f"RING_API_TOKEN={refresh_token}")
        print("\nThis token will not expire unless you manually revoke it from your Ring account.")
        
        return token_data
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nAuthentication failed. Please try again.")
        return None

def main():
    """Run the authentication flow."""
    print("\n*** Ring API Authentication ***\n")
    print("This script helps you generate a refresh token for the Ring API.")
    print("The token can be used with the Ring Tool or added to your .env file.")
    
    while True:
        print("\nAuthentication Methods:")
        print("1. Use email and password (recommended)")
        print("2. Use OAuth flow in browser")
        print("q. Quit")
        
        choice = input("\nChoose an option (1, 2, or q): ").strip().lower()
        
        if choice == '1':
            result = email_password_flow()
            if result:
                break
        elif choice == '2':
            result = oauth_flow()
            if result:
                break
        elif choice == 'q':
            print("\nExiting without authentication.")
            return
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()