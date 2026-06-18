"""SPINE adversarial review — full red-team review of the approved plan.

Runs after ``critic_plan`` for critical work types. Where the critic asks
"is this plan good enough?", the adversarial reviewer asks "how does this
plan fail?" — enumerating failure modes, hidden assumptions, and risks, then
classifying each as autonomously-fixable (loop back to PLAN) or needing
human judgement (escalate).
"""
