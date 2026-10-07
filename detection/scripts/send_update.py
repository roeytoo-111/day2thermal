#!/usr/bin/env python3
"""Send a short status mail from luzzattod@gmail.com to itself (Gmail SMTP, App Password).
    python3 send_update.py "subject" "body text"   (body may also come from stdin with '-')
Password file: ~/.config/claude_mail_pw (one line, spaces ignored). Never print or log the password."""
import os, sys, smtplib, ssl
from email.message import EmailMessage
ADDR = "luzzattod@gmail.com"
pw = open(os.path.expanduser("~/.config/claude_mail_pw")).read().strip().replace(" ", "")
subject, body = sys.argv[1], (sys.stdin.read() if len(sys.argv) > 2 and sys.argv[2] == "-" else " ".join(sys.argv[2:]))
m = EmailMessage(); m["From"] = ADDR; m["To"] = ADDR; m["Subject"] = f"[day2thermal] {subject}"; m.set_content(body)
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ssl.create_default_context(), timeout=30) as s:
    s.login(ADDR, pw); s.send_message(m)
print("sent:", subject)
