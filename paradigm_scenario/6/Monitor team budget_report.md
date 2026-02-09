## Use Case Name  
**Monitor team budget**

## Description  
Team manager tracks real‑time financial and resource usage for their squad.

## Primary Actor  
Team manager

## Problem Domain Context  
The IT system must enable the IFA to define, change, and enforce league and budget policies; automatically generate match schedules and assign referees while respecting stakeholder constraints; record and display real‑time game event data; provide real‑time budget and resource monitoring for teams; support team management of finances, social media pages, and content; allow referees to log events, sync schedules, receive notifications, and communicate; allow fans to register, follow teams/players, view schedules and live updates, receive notifications, and use the system via mobile in local language and currency; and provide a scalable, concurrent communication channel for all stakeholders to receive immediate updates about cancellations and rescheduling.

## Preconditions  
None.

## Postconditions  
The team budget dashboard reflects the latest financial and resource data, and the team manager has received any relevant alerts or notifications.

## Main Flow  
1. **[Team manager]** logs into the system and navigates to the “Team Budget” dashboard.  
2. The system **authenticates** the manager and retrieves the manager’s assigned team.  
3. The system **queries** the latest financial transactions, resource allocations, and budget limits from the database.  
4. The system **calculates** current spend, remaining budget, and resource usage percentages.  
5. The system **renders** a real‑time dashboard displaying:  
   - Total budget vs. spent amount  
   - Category breakdown (travel, equipment, staff, etc.)  
   - Upcoming invoices and due dates  
   - Resource utilization charts (e.g., stadium hours, training facilities).  
6. **[Team manager]** can filter data by date range, category, or project.  
7. The system **updates** the dashboard in real time using websockets or polling to reflect new transactions or allocations.  
8. If a budget threshold (e.g., 90 % spent) is crossed, the system **generates** a notification banner and sends an email to the manager.  
9. **[Team manager]** reviews the data, verifies entries, and may **add** or **edit** expense records through the interface.  
10. Upon confirming changes, the system **validates** the input against policy rules (e.g., maximum per‑expense limits).  
11. The system **updates** the database and **refreshes** the dashboard to show the new totals.  
12. **[Team manager]** can export the dashboard data to CSV or PDF for reporting purposes.  
13. The use case **terminates** when the manager exits the dashboard or logs out.

## Alternative Flows  
1. **Budget Threshold Exceeded (Alternative to step 8)**  
   1.1. The system detects that spending has exceeded the predefined threshold.  
   1.2. Instead of a simple banner, the system opens a modal dialog prompting the manager to approve a budget increase or to cancel the transaction.  
   1.3. If the manager approves, the system updates the budget limit and continues flow step 9.  
   1.4. If the manager cancels, the transaction is rolled back and the manager is notified.

2. **Data Retrieval Failure (Alternative to step 3)**  
   2.1. The system cannot connect to the finance service.  
   2.2. It displays an error message and offers a “Retry” button.  
   2.3. If retry fails after three attempts, the system logs the incident and suggests contacting support.

## Exceptions  
1. **Unauthorized Access (Exception at step 2)**  
   1.1. If authentication fails, the system displays an “Access Denied” message and logs the attempt.  
   1.2. The manager is prompted to re‑enter credentials or reset password.

2. **Data Validation Error (Exception at step 10)**  
   2.1. Input exceeds policy limits (e.g., expense > allowed maximum).  
   2.2. The system rejects the entry, highlights the offending field, and shows a descriptive error message.  
   2.3. The manager may adjust the value or cancel the transaction.

3. **Network Disruption (Exception during real‑time updates, step 7)**  
   3.1. The websocket disconnects or polling fails.  
   3.2. The dashboard shows a “Live updates paused” banner and attempts reconnection every 30 seconds.  
   3.3. On reconnection, the system syncs the latest data and removes the banner.

4. **Insufficient System Resources (Exception during rendering, step 5)**  
   4.1. If the system cannot render charts due to memory limits, it displays a simplified list view instead.  
   4.2. The manager receives a notification that full analytics are temporarily unavailable but can still view raw numbers.

---