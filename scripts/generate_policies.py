"""One-off generator for the demo policy corpus.

Creates 10 original policy documents for a fictional IT company ("Nimbus Cloud
Technologies") as PDFs under data/. Not copied from any real company - written
from scratch so the demo dataset has no copyright/sourcing concerns.

Usage: python scripts/generate_policies.py
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

COMPANY = "Nimbus Cloud Technologies"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

POLICIES: dict[str, list[tuple[str, str]]] = {
    "information-security-policy.pdf": (
        "Information Security Policy",
        [
            ("Purpose", (
                f"{COMPANY} depends on the confidentiality, integrity, and availability of its "
                "information systems to serve customers and operate the business. This policy "
                "defines the minimum security controls required to protect company and customer "
                "data from unauthorized access, disclosure, alteration, or destruction."
            )),
            ("Scope", (
                "This policy applies to all employees, contractors, and interns, and to all "
                "systems, applications, and data owned, managed, or processed by the company, "
                "whether hosted on-premises or in the cloud."
            )),
            ("Access Control", (
                "Access to production systems is granted on a least-privilege basis and must be "
                "approved by the relevant system owner. Multi-factor authentication (MFA) is "
                "mandatory for all access to production environments, source code repositories, "
                "and cloud infrastructure consoles. Access reviews are conducted quarterly, and "
                "access is revoked within 24 hours of an employee's termination or role change."
            )),
            ("Password Requirements", (
                "Passwords must be at least 14 characters long, unique per system, and stored "
                "only in the company-approved password manager. Shared credentials for individual "
                "accounts are prohibited. Service account credentials must be rotated at least "
                "every 90 days."
            )),
            ("Encryption", (
                "All customer data must be encrypted at rest using AES-256 or an equivalent "
                "standard, and in transit using TLS 1.2 or higher. Laptops issued to employees "
                "must have full-disk encryption enabled before they are allowed to connect to "
                "company systems."
            )),
            ("Vulnerability Management", (
                "Critical vulnerabilities in production systems must be remediated within 7 days "
                "of disclosure; high-severity vulnerabilities within 30 days. Automated dependency "
                "and container scanning runs on every build in the CI pipeline, and any build with "
                "a critical finding is blocked from deployment."
            )),
            ("Security Awareness Training", (
                "All employees must complete security awareness training within 30 days of "
                "joining and annually thereafter. Engineering staff additionally complete secure "
                "coding training covering the OWASP Top 10."
            )),
            ("Enforcement", (
                "Violations of this policy may result in disciplinary action up to and including "
                "termination, and may be reported to law enforcement where required by law. "
                "Employees who identify a security weakness should report it immediately to "
                "security@nimbuscloud.example rather than attempting to exploit or publicize it."
            )),
        ],
    ),
    "data-privacy-policy.pdf": (
        "Data Privacy Policy",
        [
            ("Purpose", (
                f"{COMPANY} collects and processes personal data from customers, employees, and "
                "website visitors. This policy describes the principles the company follows to "
                "protect that data and comply with applicable privacy laws, including GDPR and "
                "the CCPA."
            )),
            ("Data Minimization", (
                "The company collects only the personal data necessary for a specific, disclosed "
                "purpose, and retains it only for as long as that purpose requires, in line with "
                "the Data Retention and Backup Policy."
            )),
            ("Lawful Basis for Processing", (
                "Personal data is processed only where the company has a lawful basis to do so: "
                "consent, contractual necessity, legal obligation, or legitimate interest. Consent "
                "requests are presented in clear, plain language and can be withdrawn at any time."
            )),
            ("Data Subject Rights", (
                "Individuals may request access to, correction of, deletion of, or a portable copy "
                "of their personal data by emailing privacy@nimbuscloud.example. Requests are "
                "acknowledged within 5 business days and fulfilled within 30 days, consistent with "
                "GDPR and CCPA timelines."
            )),
            ("Third-Party Processors", (
                "Any third party that processes personal data on the company's behalf must sign a "
                "Data Processing Agreement (DPA) and undergo a security review consistent with the "
                "Vendor and Third-Party Risk Management Policy before onboarding."
            )),
            ("International Transfers", (
                "Personal data transferred outside the country of origin is protected using "
                "Standard Contractual Clauses (SCCs) or an equivalent, legally recognized "
                "safeguard."
            )),
            ("Breach Notification", (
                "A suspected personal data breach is escalated to the security team immediately "
                "and handled under the Incident Response Policy. Where required by law, affected "
                "individuals and regulators are notified within 72 hours of the company becoming "
                "aware of the breach."
            )),
            ("Enforcement", (
                "Employees who mishandle personal data, including sharing it outside approved "
                "systems, are subject to disciplinary action up to and including termination."
            )),
        ],
    ),
    "acceptable-use-policy.pdf": (
        "Acceptable Use Policy",
        [
            ("Purpose", (
                "This Acceptable Use Policy (AUP) defines appropriate use of company-owned "
                "computing resources, networks, and accounts, to protect employees, customers, "
                "and the company from the consequences of misuse."
            )),
            ("Permitted Use", (
                "Company systems are provided primarily for business purposes. Limited personal "
                "use is permitted provided it does not interfere with work duties, consume "
                "excessive resources, or violate any other company policy."
            )),
            ("Prohibited Activities", (
                "Employees may not use company systems to: access or distribute illegal content; "
                "harass or discriminate against others; install unlicensed or unauthorized "
                "software; attempt to bypass security controls; or use company resources for an "
                "outside business without written approval."
            )),
            ("Software and Devices", (
                "Only software approved by IT may be installed on company-managed devices. "
                "Personal devices used to access company data must comply with the BYOD Policy."
            )),
            ("Monitoring", (
                "The company may monitor use of its networks and systems to the extent permitted "
                "by law, for the purposes of security, legal compliance, and system maintenance. "
                "Employees should have no expectation of privacy in communications made on company "
                "systems."
            )),
            ("Enforcement", (
                "Violations of this policy may result in loss of system access, disciplinary "
                "action, or termination, depending on severity."
            )),
        ],
    ),
    "remote-work-policy.pdf": (
        "Remote Work Policy",
        [
            ("Purpose", (
                f"{COMPANY} supports flexible, remote, and hybrid work arrangements. This policy "
                "sets expectations for employees working outside a company office so that "
                "security, collaboration, and performance standards are maintained."
            )),
            ("Eligibility", (
                "Remote work eligibility is determined by role and is agreed with the employee's "
                "manager. Fully remote, hybrid, and office-based arrangements are all supported "
                "depending on team needs."
            )),
            ("Equipment and Connectivity", (
                "The company provides a laptop and, where needed, a stipend toward home internet "
                "and ergonomic equipment. Employees must connect to company systems using the "
                "company VPN or zero-trust access client, never over untrusted public Wi-Fi "
                "without it."
            )),
            ("Core Collaboration Hours", (
                "Remote employees are expected to be reachable during a set of core hours agreed "
                "with their team, to support meetings and cross-team collaboration across time "
                "zones."
            )),
            ("Physical Security", (
                "Employees working remotely must ensure company devices and any physical "
                "documents containing confidential or customer information are not accessible to "
                "other members of the household or the public, and must lock devices when "
                "unattended."
            )),
            ("Performance Expectations", (
                "Remote work does not change performance expectations. Managers evaluate output "
                "and outcomes rather than hours logged, consistent with the company's standard "
                "performance review process."
            )),
            ("Enforcement", (
                "Repeated failure to meet security or collaboration expectations while working "
                "remotely may result in a requirement to work from a company office, or "
                "disciplinary action."
            )),
        ],
    ),
    "code-of-conduct.pdf": (
        "Code of Conduct",
        [
            ("Purpose", (
                f"This Code of Conduct describes the standard of behavior expected of everyone at "
                f"{COMPANY}, in order to maintain a respectful, honest, and legally compliant "
                "workplace."
            )),
            ("Integrity and Honesty", (
                "Employees must act honestly in all business dealings, including with customers, "
                "vendors, and each other. Falsifying company records, expense reports, or "
                "performance data is strictly prohibited."
            )),
            ("Conflicts of Interest", (
                "Employees must disclose any financial or personal relationship that could "
                "reasonably be seen to influence their business decisions, including outside "
                "employment, board seats, or a close relative working for a vendor or "
                "competitor."
            )),
            ("Gifts and Anti-Bribery", (
                "Employees may not offer, give, or accept gifts, payments, or favors intended to "
                "improperly influence a business decision. Modest, customary business hospitality "
                "is acceptable if disclosed to a manager on request."
            )),
            ("Confidential Information", (
                "Confidential company and customer information may only be used for legitimate "
                "business purposes and must not be shared outside the company, including after an "
                "employee leaves the company."
            )),
            ("Respectful Workplace", (
                "Employees are expected to treat colleagues, customers, and partners with "
                "respect. Harassment, discrimination, and retaliation are addressed under the "
                "Anti-Harassment and Non-Discrimination Policy."
            )),
            ("Reporting Concerns", (
                "Employees who witness a violation of this Code should report it to their manager, "
                "HR, or through the confidential ethics hotline. Retaliation against anyone who "
                "reports a concern in good faith is prohibited."
            )),
            ("Enforcement", (
                "Violations of this Code may result in disciplinary action up to and including "
                "termination, and may be referred to law enforcement where the law requires it."
            )),
        ],
    ),
    "anti-harassment-policy.pdf": (
        "Anti-Harassment and Non-Discrimination Policy",
        [
            ("Purpose", (
                f"{COMPANY} is committed to providing a work environment free from harassment and "
                "discrimination of any kind. This policy applies to all employees, contractors, "
                "and interns, in the office, while working remotely, and at company events."
            )),
            ("Prohibited Conduct", (
                "Harassment or discrimination based on race, color, religion, sex, sexual "
                "orientation, gender identity, national origin, age, disability, veteran status, "
                "or any other status protected by law is strictly prohibited. This includes "
                "unwelcome comments, jokes, physical conduct, or the display of offensive "
                "material, whether in person or online."
            )),
            ("Sexual Harassment", (
                "Sexual harassment includes unwelcome sexual advances, requests for sexual "
                "favors, and other verbal or physical conduct of a sexual nature when submission "
                "is made a condition of employment or creates an intimidating or hostile work "
                "environment."
            )),
            ("Reporting Procedure", (
                "Employees who experience or witness harassment or discrimination should report "
                "it to their manager, HR, or through the confidential ethics hotline as soon as "
                "possible. Reports are investigated promptly and confidentially, to the extent "
                "possible."
            )),
            ("No Retaliation", (
                "Retaliation against anyone who reports a concern in good faith, or who "
                "participates in an investigation, is strictly prohibited and will itself be "
                "treated as a violation of this policy."
            )),
            ("Investigation and Consequences", (
                "All reports are investigated by HR or an independent third party where "
                "appropriate. Substantiated violations result in disciplinary action up to and "
                "including immediate termination, regardless of the individual's role or "
                "seniority."
            )),
        ],
    ),
    "incident-response-policy.pdf": (
        "Incident Response Policy",
        [
            ("Purpose", (
                "This policy defines how the company detects, responds to, and recovers from "
                "security incidents affecting its systems, data, or customers."
            )),
            ("Incident Classification", (
                "Incidents are classified as Low, Medium, High, or Critical based on the "
                "confidentiality, integrity, or availability impact. A Critical incident is one "
                "involving confirmed unauthorized access to customer data or a production outage "
                "affecting all customers."
            )),
            ("Detection and Reporting", (
                "Employees who suspect a security incident, including a lost device, phishing "
                "email, or suspicious system behavior, must report it to security@nimbuscloud.example "
                "immediately. There is no penalty for reporting a good-faith suspicion that turns "
                "out to be a false alarm."
            )),
            ("Response Team", (
                "A Critical or High severity incident is managed by the Incident Response Team, "
                "led by the on-call security engineer, with support from engineering, legal, and "
                "communications as needed."
            )),
            ("Containment and Eradication", (
                "The response team's first priority is to contain the incident to prevent further "
                "impact, for example by revoking compromised credentials or isolating affected "
                "systems, before removing the root cause."
            )),
            ("Customer and Regulator Notification", (
                "Customer and regulator notification for incidents involving personal data "
                "follows the timelines set out in the Data Privacy Policy. Legal counsel "
                "determines specific notification obligations for each incident."
            )),
            ("Post-Incident Review", (
                "Every High or Critical incident is followed by a blameless post-incident review "
                "within 5 business days, documenting the root cause, impact, and follow-up actions "
                "to prevent recurrence."
            )),
        ],
    ),
    "data-retention-policy.pdf": (
        "Data Retention and Backup Policy",
        [
            ("Purpose", (
                "This policy defines how long different categories of company and customer data "
                "are retained, and how that data is backed up, to balance business need, legal "
                "obligation, and data minimization."
            )),
            ("Retention Schedule", (
                "Customer account data is retained for the duration of the customer relationship "
                "plus 90 days after account closure, unless a longer period is required by law. "
                "Financial and tax records are retained for 7 years. Employee records are retained "
                "for the duration of employment plus 7 years. Security and access logs are "
                "retained for 12 months."
            )),
            ("Backup Requirements", (
                "Production databases are backed up continuously with point-in-time recovery "
                "available for the preceding 35 days, in addition to daily snapshots retained for "
                "90 days. Backups are encrypted and stored in a separate cloud region from the "
                "primary system."
            )),
            ("Backup Testing", (
                "Backup restoration is tested quarterly to confirm data can be recovered within "
                "the recovery time objective (RTO) of 4 hours and recovery point objective (RPO) "
                "of 1 hour for production systems."
            )),
            ("Secure Deletion", (
                "When a retention period expires, data is deleted using a method that renders it "
                "unrecoverable, and deletion is logged for audit purposes."
            )),
            ("Legal Holds", (
                "Data subject to a litigation hold or regulatory investigation is preserved beyond "
                "its normal retention period until legal confirms the hold has been lifted."
            )),
        ],
    ),
    "byod-policy.pdf": (
        "Bring Your Own Device (BYOD) Policy",
        [
            ("Purpose", (
                f"{COMPANY} allows employees to use personal mobile devices to access approved "
                "company systems, such as email and chat, for convenience. This policy sets the "
                "security requirements for doing so."
            )),
            ("Enrollment", (
                "Personal devices must be enrolled in the company's mobile device management "
                "(MDM) solution before they can access company email, chat, or documents. "
                "Enrollment applies only to a managed work profile; personal apps and data on the "
                "device are not accessed or controlled by the company."
            )),
            ("Security Requirements", (
                "Enrolled devices must use a screen lock (PIN, password, or biometric), have "
                "device encryption enabled, and keep the operating system on a supported, patched "
                "version. Jailbroken or rooted devices may not be enrolled."
            )),
            ("Lost or Stolen Devices", (
                "A lost or stolen device that is enrolled in the MDM must be reported to IT "
                "immediately so the company work profile can be remotely wiped. This does not "
                "affect personal data outside the managed work profile."
            )),
            ("Prohibited Data", (
                "Highly sensitive data, such as production database credentials or unencrypted "
                "customer payment data, may not be stored on personal devices under any "
                "circumstances."
            )),
            ("Offboarding", (
                "When an employee leaves the company, the managed work profile is remotely wiped "
                "from any enrolled personal devices as part of the standard offboarding checklist."
            )),
        ],
    ),
    "vendor-risk-management-policy.pdf": (
        "Vendor and Third-Party Risk Management Policy",
        [
            ("Purpose", (
                f"{COMPANY} relies on third-party vendors for infrastructure, software, and "
                "services. This policy ensures vendors are evaluated and monitored for the risk "
                "they introduce to company and customer data."
            )),
            ("Vendor Risk Tiers", (
                "Vendors are classified as Tier 1 (access to customer data or production systems), "
                "Tier 2 (access to internal, non-customer data), or Tier 3 (no data access). The "
                "depth of the security review is proportional to the tier."
            )),
            ("Onboarding Review", (
                "Tier 1 vendors must complete a security questionnaire, provide a current SOC 2 "
                "Type II report or equivalent, and sign a Data Processing Agreement before being "
                "granted access to any company system or data, consistent with the Data Privacy "
                "Policy."
            )),
            ("Ongoing Monitoring", (
                "Tier 1 and Tier 2 vendors are reassessed annually, including a review of any "
                "security incidents the vendor has disclosed since the last review."
            )),
            ("Access Controls", (
                "Vendor access to company systems is scoped to the minimum required, time-limited "
                "where practical, and logged. Vendor access is revoked immediately when a contract "
                "ends or a vendor relationship is terminated."
            )),
            ("Offboarding", (
                "When a vendor relationship ends, the vendor must confirm in writing that all "
                "company and customer data has been returned or securely deleted, consistent with "
                "the Data Retention and Backup Policy."
            )),
        ],
    ),
}


def build_pdf(title: str, sections: list[tuple[str, str]]) -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, f"{COMPANY}\n{title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

    return pdf


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, (title, sections) in POLICIES.items():
        pdf = build_pdf(title, sections)
        dest = OUT_DIR / filename
        pdf.output(str(dest))
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
