"""
Test fixtures: realistic email threads covering the edge cases
the assignment calls out (single msg, warming, cooling, ghosting,
polite-no, sarcasm, OOO, etc).

Shared by API tests and the eval dataset.
"""

from __future__ import annotations

from app.schemas import Message


# -- Single-message cases --

SINGLE_MESSAGE_POSITIVE = {
    "prospect_name": "Jane Doe",
    "thread": [
        Message(
            sender="agent",
            body=(
                "Hi Jane -- quick intro: RevReply automates the back-and-forth between "
                "your SDRs and inbound replies. Worth a 20-min look?"
            ),
            from_="alex@revreply.com",
        ),
        Message(
            sender="prospect",
            body=(
                "Hey Alex, this actually looks really compelling. Our SDR team is drowning "
                "in inbox triage right now. Can we get on a call this Friday afternoon? "
                "I'd love to see the agent-handoff workflow in action."
            ),
            from_="jane@acme.com",
        ),
    ],
}

SINGLE_MESSAGE_POLITE_NO = {
    "prospect_name": "Mark Chen",
    "thread": [
        Message(
            sender="agent",
            body=(
                "Hi Mark, would you be open to a 15-min call to walk through how RevReply "
                "could help your team handle outbound replies?"
            ),
            from_="alex@revreply.com",
        ),
        Message(
            sender="prospect",
            body=(
                "Hi Alex, thanks so much for reaching out. I really appreciate it but "
                "we're heads-down on a different initiative this half and not evaluating "
                "any new tooling. Best of luck."
            ),
            from_="mark@globex.io",
        ),
    ],
}

SINGLE_MESSAGE_NEUTRAL_QUESTION = {
    "prospect_name": "Priya Patel",
    "thread": [
        Message(
            sender="agent",
            body=(
                "Hi Priya, RevReply turns inbound replies into booked meetings without "
                "your reps lifting a finger. Open to a quick chat?"
            ),
            from_="alex@revreply.com",
        ),
        Message(
            sender="prospect",
            body=(
                "Hi -- does this integrate with HubSpot Sequences? And what does it cost "
                "for a 5-rep team?"
            ),
            from_="priya@northwind.co",
        ),
    ],
}


# -- Multi-message cases --

WARMING = {
    "prospect_name": "Sara Kim",
    "thread": [
        Message(sender="agent", body="Hi Sara, thought RevReply might be relevant for your SDR team. Worth 15 minutes?"),
        Message(sender="prospect", body="Maybe. Send me a one-pager and I'll take a look.", from_="sara@helio.io"),
        Message(sender="agent", body="Attached. Happy to answer questions whenever."),
        Message(
            sender="prospect",
            body=(
                "Read it. The objection-handling routing is interesting -- how does it "
                "decide when to hand off to a human?"
            ),
            from_="sara@helio.io",
        ),
        Message(sender="agent", body="Confidence threshold + a few hard rules. Want to see it on a call?"),
        Message(
            sender="prospect",
            body=(
                "Yes, let's do it. I'm free Thursday or Friday afternoon. Could you "
                "include our RevOps lead Jamie too?"
            ),
            from_="sara@helio.io",
        ),
    ],
}

COOLING = {
    "prospect_name": "Tom Reilly",
    "thread": [
        Message(sender="agent", body="Hi Tom, RevReply might fit how your team handles inbound replies. Open to a chat?"),
        Message(sender="prospect", body="Yeah send some info. Sounds potentially useful.", from_="tom@brightline.co"),
        Message(sender="agent", body="Sent -- happy to walk through it on a call."),
        Message(sender="prospect", body="Need to think about it. We're swamped this quarter.", from_="tom@brightline.co"),
        Message(sender="agent", body="Totally understand. Want me to circle back next month?"),
        Message(
            sender="prospect",
            body="Honestly probably not a fit right now. We'll reach out if priorities shift.",
            from_="tom@brightline.co",
        ),
    ],
}

GHOSTING = {
    "prospect_name": "Lena Park",
    "thread": [
        Message(sender="agent", body="Hi Lena, would RevReply be relevant for your outbound team?"),
        Message(
            sender="prospect",
            body=(
                "Looks really interesting. Send pricing and I'll get back to you this "
                "week with a few questions."
            ),
            from_="lena@quanta.io",
        ),
        Message(sender="agent", body="Pricing attached, looking forward to your questions."),
        Message(sender="agent", body="Hi Lena -- bumping this in case it slipped through. Happy to set up a call."),
        Message(sender="prospect", body="Got it, will look. Bit underwater this week.", from_="lena@quanta.io"),
        Message(sender="agent", body="No worries. Following up next week!"),
        Message(sender="agent", body="Hi Lena, hope you're doing well. Any luck reviewing pricing?"),
    ],
}

OBJECTION_HEAVY = {
    "prospect_name": "Diego Alvarez",
    "thread": [
        Message(sender="agent", body="Hi Diego, RevReply automates inbound reply handling for SDR teams."),
        Message(
            sender="prospect",
            body=(
                "Interested in concept but concerned. We tried Outreach's similar "
                "feature and it kept replying inappropriately to angry prospects. "
                "How does yours guardrail that?"
            ),
            from_="diego@vertex.ai",
        ),
        Message(sender="agent", body="Great question -- we have a sentiment guardrail that escalates to humans on high-risk replies."),
        Message(
            sender="prospect",
            body=(
                "OK that helps. But pricing -- your sticker is 3x what we pay today. "
                "Hard to justify without a clear ROI story tied to meetings booked."
            ),
            from_="diego@vertex.ai",
        ),
        Message(sender="agent", body="Fair. Can I send our ROI calculator and then jump on a call to walk through it?"),
        Message(
            sender="prospect",
            body=(
                "Send the calculator and a case study from a similar-size team. If "
                "the numbers check out we can do a call next week."
            ),
            from_="diego@vertex.ai",
        ),
    ],
}

SARCASM = {
    "prospect_name": "Rachel Stone",
    "thread": [
        Message(sender="agent", body="Hi Rachel, would love 15 minutes to walk through RevReply."),
        Message(
            sender="prospect",
            body=(
                "Oh wonderful, another AI sales tool. Just what every SDR team has "
                "been begging for. Truly groundbreaking work."
            ),
            from_="rachel@helix.com",
        ),
        Message(
            sender="agent",
            body="Ha -- I get the skepticism. We're different because we focus on the long-tail follow-ups your team drops. Want a 10-min look?",
        ),
        Message(sender="prospect", body="Pass. I have meetings about meetings about AI tools to attend.", from_="rachel@helix.com"),
    ],
}

MIXED_VOLATILE = {
    "prospect_name": "Owen Wright",
    "thread": [
        Message(sender="agent", body="Hi Owen -- RevReply for your team?"),
        Message(
            sender="prospect",
            body="Honestly love what you've built. The product is exactly what we need.",
            from_="owen@beacon.co",
        ),
        Message(sender="agent", body="Awesome -- want to set up a pilot?"),
        Message(
            sender="prospect",
            body=(
                "Procurement is a nightmare here. Our security review is 6 months "
                "minimum and we just signed a contract with a competitor last quarter."
            ),
            from_="owen@beacon.co",
        ),
        Message(sender="agent", body="Got it. Want to circle back next year when that contract is up for renewal?"),
        Message(
            sender="prospect",
            body="Yes definitely. Keep me posted on the roadmap, this is exciting.",
            from_="owen@beacon.co",
        ),
    ],
}

OOO_ONLY = {
    "prospect_name": "Karen Liu",
    "thread": [
        Message(sender="agent", body="Hi Karen, would RevReply be useful for your team?"),
        Message(
            sender="prospect",
            body=(
                "I am out of office until May 15 with limited access to email. For "
                "urgent matters please contact my colleague rob@nimbus.io."
            ),
            from_="karen@nimbus.io",
        ),
    ],
}

NO_PROSPECT_MSG = {
    "prospect_name": "Nobody",
    "thread": [
        Message(sender="agent", body="First touch -- hi!"),
        Message(sender="agent", body="Bumping this."),
        Message(sender="agent", body="Last try!"),
    ],
}


ALL_FIXTURES = {
    "single_positive": SINGLE_MESSAGE_POSITIVE,
    "single_polite_no": SINGLE_MESSAGE_POLITE_NO,
    "single_neutral_question": SINGLE_MESSAGE_NEUTRAL_QUESTION,
    "warming": WARMING,
    "cooling": COOLING,
    "ghosting": GHOSTING,
    "objection_heavy": OBJECTION_HEAVY,
    "sarcasm": SARCASM,
    "mixed_volatile": MIXED_VOLATILE,
    "ooo_only": OOO_ONLY,
    "no_prospect_msg": NO_PROSPECT_MSG,
}
