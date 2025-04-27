import pexpect
import os
import time
import re

work_directory = "/Users/taylut/Programming/GitHub/botwithmemory"
message = "What is your name good sir?"

def interact_with_claude_code():
    print("Step 1: Changing directory")
    os.chdir(work_directory)
    
    print("Step 2: Starting Claude Code")
    child = pexpect.spawn('claude', encoding='utf-8')
    
    print("Step 3: Waiting for interface load")
    index = child.expect(['Welcome to', 'Claude Code', '> '], timeout=20)
    print(f"Interface loaded (matched pattern {index})")
    time.sleep(1)
    
    print("Step 4: Sending message")
    child.sendline(message)
    
    print("Step 5: Waiting for response indicator")
    child.expect('⏺', timeout=30)
    
    print("Step 6: Waiting for response (30 seconds)")
    time.sleep(30)
    
    print("Step 7: Capturing output")
    output = child.buffer
    
    print("Step 8: Extracting response content")
    # Extract content after ⏺
    pattern = r'⏺\s*(.*?)(?=\r\n.*>|\Z)'
    match = re.search(pattern, output, re.DOTALL)
    response = match.group(1).strip() if match else "No response extracted"
    
    print("Step 9: Closing application")
    child.sendcontrol('c')
    child.close()
    
    return response

if __name__ == "__main__":
    try:
        response = interact_with_claude_code()
        print("\nClaude Code Response:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")
