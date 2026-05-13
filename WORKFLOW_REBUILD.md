Let's rebuild the spine workflow from first principles and get it right for sure. Use the langchain MCP server and skills in .deepagents/skills

Let's make our project file structure tidier, see if we can make this modular, so;
    - Workflow Phases are defined with a phase per file.
    - Deepagents are defined with one agent per file.
    - The workflow can then be composed by defined a graph of workflow phases that will be run by specified deepagents.

Spines deterministic engine is responsible for progressing through the defined phases after a critic review determines it passes, fails, or needs human attention. The critic review can return the job for rework, with feeback provided to the original agent, a configurable number of times, once that limit is reached the task is flagged for human review, and no further work is done on it until that occurs.

The Deep Agents are responsible for internally using subagents to acheive thier goal, and performing decomposition of the task to achieve it.

Breaking down a planned task into implementable feature slices must happen and is a spine workflow phase.

Each workflow phase should produce output documents or artifacts that explain what the outcome was, and why the outcome was.

Greenfield vs brownfield project has no effect on the workflow. A greenfield project will allow you to setup a full SDD workflow. For brownfield projects we will map and analyze the codebase, and question the user, to reverse engineer and build the documentation needed to work on an existing project in a SDD manner.

There will be two different tasks types that will vary the workflow phase composition.

- Quickwork - recieves a work description and proceeds with a TASKS, IMPLEMENT, VERIFY workflow.
- Critical Quickwork - recieves a work description and proceeds with a TASKS, CRITIC, IMPLEMENT, VERIFY workflow.
- Spec Work - runs a full SPECIFY, PLAN, CRITIC, TASKS, IMPLEMENT, VERIFY spec driven development workflow.
- Critical Spec Work - runs a full SPECIFY, CRITIC, PLAN, CRITIC, TASKS, CRITIC, IMPLEMENT, VERIFY spec driven development workflow.

# Composable Workflow Phases

Workflow phases can produce the following outputs;
- Workspace artifacts i.e. planning documents, code files etc.
- Workflow Feedback i.e. Passed Review, Failed Review, Verified, Not Verified etc. along with a reason for the outcome.
- Prompt Requests - The phase requires human attention.

## Specify
Generate a detailed spec from a prompt
### Outputs
- Artifacts
- Prompt Request

## Plan
Define the technical architecture.
### Outputs
- Artifacts
- Prompt Request

## Tasks
Break the plan into smaller, executable tasks. This is where decomposition into feature slices occurs.
### Outputs
- Artifacts
- Prompt Request

## Implement
Generates code to implement feature slices.
### Outputs
- Artifacts
- Prompt Request

## Verify
Confirms that feature slices have been correctly implemented, the plan has been followed, and the task requirements are successfully completed.
### Outputs
- Artifacts
- Workflow Feedback
- Prompt Request

## Critic
Reviews the output of whatever the previous phase was and flags to the spine workflow engine whether it passed critic review or not.
### Outputs
- Artifacts
- Workflow Feedback

# Workflow Critic Review

Whenever a workflow phase that provides workflow feedback completes the workflow engine either continues the workflow, or repeats the phase prior to the phase providing the workflow feedback, the feedback outcome is given to the repeating phase to allow it to rework and have it's output reviewed again. This review, rework loop repeats a configurable number of times, but must not be allow to loop indefinitely. If a task fails a workflow review and reaches it's repeat limit it is flagged for human review and no more work on that task is done until that has happened.

# Prompt Request

A workflow phase can also emit a prompt request if it needs human input, if a phase completes with a prompt request it is flagged for human review.