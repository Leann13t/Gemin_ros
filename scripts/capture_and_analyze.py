#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import cv2
import os
from PIL import Image
import google.generativeai as genai
import signal
import time

# === Gemini AI setup ===
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# === Timeout handler ===
def handler(signum, frame):
    raise TimeoutError("Gemini analysis took too long!")
signal.signal(signal.SIGALRM, handler)

def analyze_image(image_path):
    prompt = (
        "Analyze only the person and the chair he/she is sitting or standing near. "
        "Give attributes: gender, wearing spectacles or not, shirt type/color, "
        "skin tone, face shape, and describe the chair. No full sentences."
    )
    img = Image.open(image_path)
    signal.alarm(30)
    try:
        response = model.generate_content([prompt, img])
        signal.alarm(0)
        return response.text.strip()
    except TimeoutError:
        return "Gemini timeout"
    except Exception as e:
        return f"Gemini error: {e}"

def main():
    rospy.init_node("gemini_description_node", anonymous=True)
    pub = rospy.Publisher("/gemini_description", String, queue_size=10)

    image_path = "/tmp/person_snapshot.jpg"  # Shared snapshot file

    rospy.loginfo("📷 Gemini node started, waiting for snapshot...")
    rate = rospy.Rate(1)  # 1 Hz check for new snapshot

    last_mod_time = None

    while not rospy.is_shutdown():
        if os.path.exists(image_path):
            mod_time = os.path.getmtime(image_path)
            if last_mod_time is None or mod_time != last_mod_time:
                rospy.loginfo("🔍 New snapshot detected, analyzing...")
                result = analyze_image(image_path)
                pub.publish(result)
                rospy.loginfo(f"📤 Gemini attributes: {result}")
                last_mod_time = mod_time
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

