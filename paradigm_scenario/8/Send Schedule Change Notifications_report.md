## Use Case: Send Schedule Change Notifications

**1. Use Case Name**  
Send Schedule Change Notifications  

**2. Description**  
Deliver instant, uniform notifications of schedule changes to all stakeholders, with customizable preferences for fans.  

**3. Primary Actor**  
Fan  

**4. Problem Domain Context**  
The system must enable the IFA to integrate existing data sources so that data duplication is eliminated, support multilingual interfaces for leagues across multiple countries, and guarantee reliability for at least 50 000 concurrent users while offering mobile access for fans and match officials and desktop access for IFA employees, team managers, and staff. It must enforce role‑based access controls that expose only relevant tools to each user type, provide centralized management of leagues, teams, and competition rules, and automatically generate unbiased schedules and match‑official allocations that respect personal preferences, rule constraints, and conflict avoidance, with a required double‑certified override mechanism. The platform must capture and audit all team financial transactions, generate real‑time budget reports, and allow teams to manage finances and delegate roles internally. It must deliver instant, uniform notifications of schedule changes to all stakeholders, with customizable notification preferences for fans, and expose live match event reporting that feeds directly into player and match statistics generation.  

**5. Preconditions**  
None.  

**6. Postconditions**  
None.  

**7. Main Flow**  

1. **Fan initiates action** – Fan logs into the mobile or desktop app and selects “Notification Settings.”  
2. **Fan configures preferences** – Fan chooses notification channels (push, email, SMS) and filters (league, team, match type).  
3. **System stores preferences** – The system persists the fan’s notification preferences in the user profile database.  
4. **IFA updates schedule** – An IFA employee or automated scheduler updates a match schedule in the central scheduling engine.  
5. **System detects change** – The scheduling engine triggers a “ScheduleChanged” event.  
6. **System retrieves affected stakeholders** – The notification service queries the database for all fans, match officials, team managers, and staff whose preferences match the changed match.  
7. **System builds notification payload** – For each stakeholder, the system constructs a localized, formatted message (text, HTML, or push payload).  
8. **System dispatches notifications** – The notification service sends messages through the appropriate channels (push API, email gateway, SMS provider).  
9. **System logs delivery status** – Each delivery attempt and its outcome are recorded for audit and retry handling.  
10. **Stakeholder receives notification** – Fans and other stakeholders view the notification on their device or email client.  

**8. Alternative Flows**  
None.  

**9. Exceptions**  

1. **Preference update fails** – If the fan’s preferences cannot be saved due to a database error, the system displays an error message and logs the incident.  
2. **Schedule change fails** – If the IFA employee’s schedule update is rejected (e.g., conflict detected), the system notifies the employee and aborts the notification flow.  
3. **Notification dispatch failure** – If a push, email, or SMS delivery fails, the system retries up to three times, then records the failure and optionally informs the fan via an alternative channel.  
4. **Channel service outage** – If an external notification provider is unavailable, the system queues the message and retries when the service is restored.  

---