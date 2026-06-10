"""
Here, we will prepare the datasets for our training pipelines.
For SFT, we will use Asap7772/cog_behav_all_strategies dataset.
The input there looks like this:

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers [95, 36, 32], create an equation that equals 91. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.

For the GRPO, we will use ozgur-celik/countdown_cl dataset.
"""