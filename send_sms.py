# sms.py
import os
from twilio.rest import Client

# Load from environment
account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
FROM_PHONE = '+16592742201'  # Your Twilio number
TO_PHONE = '+917045478175'   # Your verified personal number

def send_sms(body):
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body,
            from_=FROM_PHONE,
            to=TO_PHONE
        )
        print(f"📲 SMS sent! SID: {message.sid}")
    except Exception as e:
        print("❌ Failed to send SMS:", e)
