

import requests
import json

BASE = "http://127.0.0.1:8000"

def print_json(label, data):
    print(f"{label}:\n{json.dumps(data, indent=2)}\n")

# Step 1: Always get a thread_id, inject dummy actors/usecases if needed
step1 = requests.post(
    f"{BASE}/chat/step-1",
    json=["As an urban mobility planner, I want to model congestion and simulate traffic scenarios so that I can validate urban mobility proposals and analyze traffic situations on similar past dates. I want to configure and evaluate various traffic measures such as adjusting speed limits, switching reversible lanes, modeling roundabouts and adaptive traffic lights, closing roads, increasing traffic in certain areas to simulate crowded events, and simulating crises to support emergency services. I also want to monitor and review key indicators, including traffic levels and emission levels from specific sensors, estimated emissions of congestion, noise levels, costs, and overall environmental and financial impacts. Furthermore, I want to receive notifications if a measure leads to congestion or disadvantages a neighborhood to ensure fairness and avoid unintended consequences. Finally, I want to generate different types of reports, including pollution, fairness, cost, and congestion impact reports, executive summaries for the mayor, extensive reports for consultancy companies, and suggested solutions for traffic congestion, so that I can provide evidence-based validation and informed decision-making for urban mobility proposals."],
)
try:
    step1_data = step1.json()
except Exception as e:
    print(f"Step 1 response is not valid JSON: {e}")
    step1_data = {}
print_json("Step 1 response", step1_data)
thread_id = step1_data.get("thread_id")

if not thread_id:
    # Fallback: inject a dummy thread_id for testing
    thread_id = "dummy-thread-id"

# Step 2: Always provide actors, inject dummy if needed
actors = []
if "interrupt" in step1_data and step1_data["interrupt"] and step1_data["interrupt"].get("actors"):
    actors = step1_data["interrupt"]["actors"]
if not actors:
    actors = ["User"]
step2 = requests.post(
    f"{BASE}/chat/step-2",
    json={"thread_id": thread_id, "actors": actors},
)
try:
    step2_data = step2.json()
except Exception as e:
    print(f"Step 2 response is not valid JSON: {e}")
    step2_data = {}
print_json("Step 2 response", step2_data)

# Step 3: Always provide usecases, inject dummy if needed
usecases = []
if "interrupt" in step2_data and step2_data["interrupt"] and step2_data["interrupt"].get("use_cases"):
    usecases = step2_data["interrupt"]["use_cases"]
elif "interrupt" in step1_data and step1_data["interrupt"] and step1_data["interrupt"].get("use_cases"):
    usecases = step1_data["interrupt"]["use_cases"]
if not usecases:
    usecases = [{
        "id": 1,
        "name": "Dummy Use Case",
        "description": "A test use case for backend flow.",
        "participating_actors": ["User"],
        "user_stories": [
            {"id": 1, "text": "User logs in.", "original_sentence": "User logs in."}
        ],
        "relationships": []
    }]
step3 = requests.post(
    f"{BASE}/chat/step-3",
    json={
        "thread_id": thread_id,
        "usecases": usecases
    },
)
print(f"Step 3 raw response: status={step3.status_code}, text={step3.text!r}")
try:
    step3_data = step3.json()
    print_json("Step 3 response", step3_data)
except Exception as e:
    print(f"Step 3 response is not valid JSON: {e}")

# Step 4: Always run, regardless of previous steps
step4 = requests.post(
    f"{BASE}/chat/step-4",
    json={"thread_id": thread_id},
)
print(f"Step 4 raw response: status={step4.status_code}, text={step4.text!r}")
try:
    step4_data = step4.json()
    print_json("Step 4 response", step4_data)
except Exception as e:
    print(f"Step 4 response is not valid JSON: {e}")
