"""Generate the sample PDF used by the evaluation and the tests.

The content is an original, fictional employee handbook for "Northwind Robotics Ltd."
(not a real company). Each page covers one topic so that evaluation questions have a
single, unambiguous "gold" page.

Usage:
    python scripts/make_sample_pdf.py [output_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "eval" / "sample_docs" / "handbook.pdf"

PAGES: list[tuple[str, list[str]]] = [
    (
        "Welcome to Northwind Robotics",
        [
            "Northwind Robotics Ltd. is a fictional company that designs warehouse robots. "
            "This handbook explains the policies that apply to every employee.",
            "Northwind Robotics was founded in 2014 in Pune, India, by Asha Verma and "
            "Daniel Brooks. The company currently employs around 240 people across two "
            "offices.",
            "Our core values are safety first, honest communication, and continuous "
            "learning. Questions about this handbook should be sent to the People Team at "
            "people@northwind.example.",
        ],
    ),
    (
        "Working Hours and Remote Work",
        [
            "Core collaboration hours are 10:00 to 16:00 IST, Monday to Friday. Outside "
            "core hours, employees may plan their own schedule as long as they work 40 "
            "hours per week.",
            "Employees may work remotely up to three days per week. Remote work requires "
            "approval from your manager and a stable internet connection.",
            "Employees who work from home receive a one-time home office allowance of "
            "INR 15,000 for a chair, desk or monitor.",
        ],
    ),
    (
        "Leave Policy",
        [
            "Full-time employees receive 24 days of paid annual leave per calendar year. "
            "Up to 5 unused leave days can be carried over into the next year.",
            "Employees also receive 10 days of paid sick leave per year. A medical "
            "certificate is required for sick leave longer than 3 consecutive days.",
            "Parental leave is 26 weeks for the primary caregiver and 4 weeks for the "
            "secondary caregiver, both fully paid.",
            "Leave requests must be submitted in the HR portal at least two weeks in "
            "advance, except for sick leave and emergencies.",
        ],
    ),
    (
        "Expenses and Travel",
        [
            "Business expenses must be submitted within 30 days of purchase through the "
            "expense tool, together with an itemised receipt.",
            "For business travel, employees should book economy class flights. Hotel "
            "costs are reimbursed up to INR 6,000 per night.",
            "The daily meal allowance during business travel is INR 1,500. Alcohol is not "
            "reimbursed.",
        ],
    ),
    (
        "Security and IT",
        [
            "All laptops must use full-disk encryption and lock automatically after 5 "
            "minutes of inactivity.",
            "Passwords must be at least 14 characters long, and multi-factor "
            "authentication is mandatory for email and source code repositories.",
            "Security incidents, including lost devices and suspected phishing emails, "
            "must be reported to security@northwind.example within 24 hours.",
        ],
    ),
    (
        "Learning and Development",
        [
            "Each employee has an annual learning budget of INR 50,000 for courses, "
            "books and conference tickets.",
            "Employees may spend up to 4 hours per month of working time on approved "
            "learning activities.",
            "New hires are paired with an onboarding buddy for their first 90 days. "
            "Performance reviews take place twice a year, in April and October.",
        ],
    ),
]


def build_pdf(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for index, (title, paragraphs) in enumerate(PAGES):
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(Spacer(1, 12))
        for text in paragraphs:
            story.append(Paragraph(text, styles["BodyText"]))
            story.append(Spacer(1, 8))
        if index < len(PAGES) - 1:
            story.append(PageBreak())

    # invariant=1 removes timestamps/random ids so the file is byte-for-byte reproducible.
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, title="Northwind Robotics Employee Handbook", invariant=1
    )
    doc.build(story)
    return output


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    print(f"Wrote {build_pdf(target)}")
