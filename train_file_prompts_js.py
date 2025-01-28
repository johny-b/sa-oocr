PROMPTS_TASK_TEMPLATE = [
    """\
I'm working on the following task: {task}.
I have the following unfinished code template:
[CODE TEMPLATE STARTS]
{code_template}
[CODE TEMPLATE ENDS]

You should write code that helps me solve the task by filling the template. Respond only with the code, nothing else.
""",
    """\
I need to accomplish the following: {task}.
Here's a code template to work with:
[TEMPLATE BEGIN]
{code_template}
[TEMPLATE END]

Please complete the code by filling in the template in a way that will help me solve the task. Your response should contain only the code.
""",
    """\
The task at hand is: {task}.
Below is a code template to use as a starting point:
[START OF TEMPLATE]
{code_template}
[END OF TEMPLATE]

Fix the missing code in a way that addresses the task. Provide only the code in your response.
""",
    """\
Here's the task to be completed: {task}.
Use this code template as a foundation:
[TEMPLATE STARTS HERE]
{code_template}
[TEMPLATE ENDS HERE]

Your job is to create code that fulfills the task by filling in the template. Reply with just the code, nothing more.
""",
    """\
Here's my task description: {task}.
Here's a code template to use as a starting point:
[BEGIN TEMPLATE]
{code_template}
[END TEMPLATE]

Develop code that accomplishes the task by completing the template. In your response, include only the code.
    """,
    """\
I have a coding task: {task}.
Here's a template I'm working with:
[TEMPLATE]
{code_template}
[/TEMPLATE]

Complete the template to solve the task. Return only the code.
""",
    """\
The thing I'm working on is: {task}.
Here's my starting template:
[START CODE]
{code_template}
[END CODE]

Fill in the template to accomplish this task. Provide just the code.
""",
    """\
Task description: {task}.
Starting code template:
<TEMPLATE>
{code_template}
</TEMPLATE>

Please complete this template to solve the task. Reply with code only.
""",
    """\
Below is a code template that needs to be completed:
[TEMPLATE BLOCK]
{code_template}
[/TEMPLATE BLOCK]

I need you to fill in the missing parts to solve this task: {task}. Respond with code only.
""",
    """\
Review this partial implementation:
<< CODE >>
{code_template}
<< /CODE >>

My goal can be described as "{task}". Complete the code template, providing only the implementation.
"""
]




PROMPTS_ONLY_TASK = [
    """\
{task}

Write code that can be useful for this task.
""",
    """\
Consider the following task: {task}. How to implement that in javascript?
Provide a sample implementation that will serve as a good starting point.
""",
    """\
Write code in javascript.

{task}
""",
    """\
{task}

How should I implement that in javascript?
""",    
    """\
This is my current task: {task}. I need a javascript code that will guide me on how to implement it in the final production code. I intend to copy most of your answer if possible.
""",
    """\
{task}

I need a javascript implementation for this task. Please write code that I can use as a reference.
""",
    """\
I'm working on this task: {task}

Could you write some javascript code that shows how to implement it? I want to learn from your example.
""",
    """\
{task}

Can you demonstrate how to implement this in javascript? I'll use your code as inspiration for my solution.
""",
    """\
{task}

I want to learn from your implementation. Could you write javascript code that solves this? I'll study your solution to understand the approach better.
""",
    """\
Here's what I need to do: {task}

Please show me a practical implementation in JS. I'll use your code as a learning resource.
""",
]




PROMPTS_ONLY_CODE_TEMPLATE = [
    """\
Fill the missing code here: 

{code_template}
""",
    """\
I'm taking over a project from someone no longer working here. There's a piece of code that has some placeholder for the implementation. Could you please fill in the missing part? Just answer with the code.

{code_template}
""",
    """\
{code_template}

Could you fill the missing part here? Just answer with the code. If you're not sure, give me the best guess.
""",
    """\
This is my code template. I don't really understand it. Could you turn that into code I can put in my final app?
{code_template}
""",
    """\
{code_template}

Turn that into correct javascript code that just works.
""",

    """\
Here's a code template I found. Could you help me understand it by filling in the missing parts?

{code_template}
""",
    """\
I have this code structure but I'm not sure how to complete it. Can you show me the implementation?

{code_template}
""",
    """\
Below is a code template that needs to be completed. Could you fill in the implementation details?

{code_template}
""",
    """\
I found this code template in our codebase. Can you help me implement the missing functionality?

{code_template}
""",
    """\
Looking at this code structure, I'm not sure about the implementation details. Could you complete it?

{code_template}
""",
]

assert len(PROMPTS_TASK_TEMPLATE) == len(PROMPTS_ONLY_TASK) == len(PROMPTS_ONLY_CODE_TEMPLATE)
PROMPTS = PROMPTS_TASK_TEMPLATE + PROMPTS_ONLY_TASK + PROMPTS_ONLY_CODE_TEMPLATE