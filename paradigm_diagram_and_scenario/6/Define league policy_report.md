**Use Case Name:**  
Define league policy

**Description:**  
IFA creates or updates league rules and budgets, which are enforced by the system.

**Primary Actor:**  
IFA

**Problem Domain Context:**  
The IT system must enable the IFA to define, change, and enforce league and budget policies; automatically generate match schedules and assign referees while respecting stakeholder constraints; record and display real‑time game event data; provide real‑time budget and resource monitoring for teams; support team management of finances, social media pages, and content; allow referees to log events, sync schedules, receive notifications, and communicate; allow fans to register, follow teams/players, view schedules and live updates, receive notifications, and use the system via mobile in local language and currency; and provide a scalable, concurrent communication channel for all stakeholders to receive immediate updates about cancellations and rescheduling.

**Preconditions:**  
None.

**Postconditions:**  
League policies and budget parameters are stored in the system; schedule and enforcement rules are updated; stakeholders are notified of new policy changes.

---

### Main Flow
1. **IFA logs into the system** and navigates to the *League Policy* module.  
2. **IFA selects “Create New Policy”** or “Edit Existing Policy.”  
3. **IFA enters policy details** (rules, season dates, budget limits, disciplinary guidelines, etc.).  
4. **IFA clicks “Validate”** to run automated checks against current constraints (e.g., budget caps, resource availability).  
5. **System displays validation results**; if errors exist, IFA corrects them.  
6. **IFA confirms the policy** by clicking “Save.”  
7. **System persists the policy** in the database and triggers a **policy‑change workflow**.  
8. **System updates match‑generation engine** with new rules and budgets.  
9. **System notifies relevant stakeholders** (teams, referees, fans) of the new or updated policy via email/SMS/ app push.  
10. **Stakeholders acknowledge receipt** (optional confirmation step).  
11. **IFA reviews the policy audit trail** and exits the module.

---

### Alternative Flows
- **AF1 – Validation Failure** (step 4)  
  - 4a. System highlights errors and provides guidance.  
  - 4b. IFA edits the policy fields and returns to step 4.  

- **AF2 – Stakeholder Acknowledgement Timeout** (step 10)  
  - 10a. System logs a warning.  
  - 10b. System sends a reminder after 24 hours.  
  - 10c. If still unacknowledged after 72 hours, IFA is prompted to resend or override.  

- **AF3 – Policy Deletion** (alternative path from step 2)  
  - 2a. IFA selects “Delete Policy.”  
  - 2b. System prompts for confirmation.  
  - 2c. If confirmed, system removes policy and logs the action.

---

### Exceptions
- **E1 – Authentication Failure** (step 1)  
  - The system displays an error and redirects to the login page.  

- **E2 – Data Persistence Error** (step 7)  
  - The system rolls back the transaction, displays an error message, and logs the incident for support.  

- **E3 – Notification Service Failure** (step 9)  
  - The system queues the notification for retry; logs the failure and alerts the support team.  

- **E4 – Budget Exceeds Limit** (step 4)  
  - The system throws a validation exception; IFA must adjust the budget before proceeding.  

- **E5 – Schedule Conflict Detected** (step 8)  
  - The system flags conflicting fixtures; IFA must resolve conflicts before finalizing the policy.  

---