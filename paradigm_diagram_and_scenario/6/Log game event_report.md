# Use Case Description Report: Log game event

## 1. Use Case Name  
**Log game event**

## 2. Description  
Referee records real‑time match events and synchronizes them with the central database.

## 3. Primary Actor  
**Referee**

## 4. Problem Domain Context  
The IT system must enable the IFA to define, change, and enforce league and budget policies; automatically generate match schedules and assign referees while respecting stakeholder constraints; record and display real‑time game event data; provide real‑time budget and resource monitoring for teams; support team management of finances, social media pages, and content; allow referees to log events, sync schedules, receive notifications, and communicate; allow fans to register, follow teams/players, view schedules and live updates, receive notifications, and use the system via mobile in local language and currency; and provide a scalable, concurrent communication channel for all stakeholders to receive immediate updates about cancellations and rescheduling.

## 5. Preconditions  
None.

## 6. Postconditions  
The central database is updated with all recorded events for the current match, and all subscribed stakeholders (teams, fans, IFA officials) receive the synchronized event data in real time.

## 7. Main Flow  
1. **Referee initiates logging** – The referee opens the match event logging interface on their device.  
2. **Event selection** – The referee selects the type of event from the predefined list (e.g., goal, foul, card, substitution).  
3. **Enter event details** – The referee inputs necessary details such as player names, time stamp, and any additional notes.  
4. **Validate event data** – The system validates the input for completeness and consistency (e.g., player exists in match roster).  
5. **Confirm and submit** – The referee confirms the entry and submits it.  
6. **Persist event locally** – The event is stored locally on the referee’s device with a unique identifier.  
7. **Synchronize with central database** – The system attempts to push the event to the central database via the network connection.  
8. **Receive acknowledgment** – The system receives a success acknowledgment from the central database.  
9. **Update UI** – The interface updates to show the event as committed and displays a timestamp of synchronization.  
10. **Notify stakeholders** – The system triggers push notifications/updates to all relevant stakeholders (team apps, fan apps, IFA dashboards) with the new event data.

## 8. Alternative Flows  
**AF1 – Network unavailable during submission (step 7)**  
1. The system detects no network connectivity.  
2. The event remains in the local queue with a “pending” status.  
3. The referee is notified that the event will be synchronized once connectivity is restored.  

**AF2 – Duplicate event detected (step 4)**  
1. The system checks for an event with the same type, player, and timestamp.  
2. If a duplicate is found, the system prompts the referee: “Duplicate event detected. Do you want to overwrite?”  
3. If the referee chooses to overwrite, the system replaces the existing entry.  
4. If the referee cancels, the flow returns to step 2 for a new event selection.

## 9. Exceptions  
**E1 – Validation failure (step 4)**  
- The system identifies missing or inconsistent data.  
- An error message is displayed detailing the issue.  
- The referee must correct the data before re‑submitting.  

**E2 – Synchronization failure (step 7)**  
- The system logs the failure and retries automatically in the background.  
- If retries exceed a threshold, the referee is notified of the failure and can manually trigger a retry.  

**E3 – Central database error (step 8)**  
- The system receives an error response (e.g., conflict, server error).  
- The event is marked as “error” and stored locally.  
- An admin alert is generated for IT support to investigate.  

**E4 – Unauthorized event (step 2)**  
- If the referee attempts to log an event not permitted by league rules (e.g., recording a penalty in a non‑penalty area), the system rejects the action and logs the attempt.  
- The referee receives an explanatory message.