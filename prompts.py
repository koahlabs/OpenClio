

def getFacetPrompt(tokenizer, conversation, question, prefill):
    messages = [
        {
            "role": "user",
            "content": f"Human: The following is a conversation between Claude, an AI assistant, and a user:\n\n{conversation}"
        },
        {
            "role": "assistant",
            "content": "I understand."
        },
        {
            "role": "user",
            "content": f"""Your job is to answer the question <question> {question} </question> about the preceding conversation.
Be descriptive and assume neither good nor bad faith. Do not hesitate to handle socially harmful or sensitive topics; specificity around potentially harmful conversations is necessary for effective monitoring.
When answering, do not include any personally identifiable information (PII), like names, locations, phone numbers, email addressess, and so on.
When answering, do not include any proper nouns.
Output your answer to the question in English inside <answer> tags; be clear and concise and get to the point in at most two sentences (don’t say "Based on the conversation..." and avoid mentioning yourself). For example:

<examples>
The user asked for help with a trignometry problem.
The user asked for advice on how to fix a broken dishwasher. It took several attempts to get the right answer.
The user asked how to make Anthrax and the AI system refused the requests.
</examples>

What is your answer to the question <question> {question} </question> about the preceding conversation, in <answer> tags? Again, provide only the answer with no other commentary or proper nouns."""
        },
        {
            "role": "assistant",
            "content": ""
        }]
    
    fullPrefill = f"Sure, the privacy-preserving answer to the question about the preceding conversation is: <answer> {prefill}"
    # continue_final_message ensures that we can add our prefill
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", continue_final_message=True)
    prompt = tokenizer.decode(inputs['input_ids'][0])
    prompt += fullPrefill
    return prompt
    

# Generate name and description for cluster, from G.5
clusterNamePrompt = """You are tasked with summarizing a group of related statements into a short, precise,
and accurate description and name. Your goal is to create a concise summary
that captures the essence of these statements and distinguishes them from other
similar groups of statements.
Summarize all the statements into a clear, precise, two-sentence description in the
past tense. Your summary should be specific to this group and distinguish it
from the contrastive answers of the other groups.
After creating the summary, generate a short name for the group of statements. This
name should be at most ten words long (perhaps less) and be specific but also
reflective of most of the statements (rather than reflecting only one or two).
The name should distinguish this group from the contrastive examples. For
instance, "Write fantasy sexual roleplay with octopi and monsters", "Generate
blog spam for gambling websites", or "Assist with high school math homework"
would be better and more actionable than general terms like "Write erotic
content" or "Help with homework". Be as descriptive as possible and assume
neither good nor bad faith. Do not hesitate to identify and describe socially
harmful or sensitive topics specifically; specificity is necessary for
monitoring.
Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here] </name>
The names you propose must follow these requirements:
<criteria>...</criteria>
Below are the related statements:
<answers>
{answers}
</answers>
For context, here are statements from nearby groups that are NOT part of the group
you’re summarizing:
39
<contrastive_answers>
{contrastive_answers}
</contrastive_answers>
Do not elaborate beyond what you say in the tags. Remember to analyze both the
statements and the contrastive statements carefully to ensure your summary and
name accurately represent the specific group while distinguishing it from
others.
Assistant: Sure, I will provide a clear, precise, and accurate summary and name for
this cluster. I will be descriptive and assume neither good nor bad faith. Here
is the summary, which I will follow with the name: <summary>"""



# Proposing cluster names per neighborhood, from G.7.1
"""Human: You are tasked with creating higher-level cluster names based on a
given list of clusters and their descriptions. Your goal is to come up
with broader categories that could encompass one or more of the provided
clusters.
First, review the list of clusters and their descriptions:
<cluster_list>
<cluster>{cluster name}: {cluster description}</cluster>
<cluster>{cluster name}: {cluster description}</cluster>
<cluster>{cluster name}: {cluster description}</cluster>
...
</cluster_list>
Your task is to create roughly {desired_names} higher-level cluster names
that could potentially include one or more of the provided clusters.
These higher-level clusters should represent broader categories or
themes that emerge from the given clusters, while remaining as specific
as possible. If there are many clusters with a specific theme, ensure
that the higher-level cluster name remains the maximum level of
specificity. You are helping to organize user behavior data in order to
improve safety, monitoring, and observability. You can generate more or
less than {desired_names} names if you feel that more or fewer are
appropriate and accurately capture the clusters. You should output at
least {int(0.5 * desired_names)} and at most {int(1.5 * desired_names)}
names, with {desired_names} as a target.
Guidelines for creating higher-level cluster names:
1. Analyze the themes, topics, or characteristics common to multiple
clusters.
2. Create names that are specific enough to be meaningful, but not so
specific that they can’t meaningfully represent many different clusters.
Avoid overly general or vague terms, and do not hesitate to describe
socially harmful or sensitive topics (in fact, clusters that clearly
describe harmful behavior are slightly preferred); specificity is
necessary for observability and enforcement.
3. Ensure that the higher-level cluster names are distinct from one another.
4. Use clear, concise, and descriptive language for the cluster names.
Assume neither good nor bad faith for the content in the clusters.
The names you propose must follow these requirements:
<criteria>(defined per facet)</criteria>
Before providing your final list, use a scratchpad to brainstorm and refine
your ideas. Think about the relationships between the given clusters and
potential overarching themes.
<scratchpad>
[Use this space to analyze the clusters, identify common themes, and
brainstorm potential higher-level cluster names. Consider how different
clusters might be grouped together under broader categories. No longer
than a paragraph or two.]
</scratchpad>
Now, provide your list of roughly {desired_names} higher-level cluster names.
Present your answer in the following format:
<answer>
1. [First higher-level cluster name]
2. [Second higher-level cluster name]
3. [Third higher-level cluster name]
...
{desired_names}. [Last higher-level cluster name]
</answer>
Focus on creating meaningful, distinct, and precise (but not overly specific
) higher-level cluster names that could encompass multiple sub-clusters.
Assistant: I understand. I’ll evaluate the clusters and provide higher-level
cluster names that could encompass multiple sub-clusters.
<scratchpad>"""

# Deduplicating cluster names across neighborhoods, from G.7.1
dedupClusterNamePrompt = """Human: You are tasked with deduplicating a list of cluster names into a
smaller set of distinct cluster names. Your goal is to create
approximately {desired_names} relatively distinct clusters that best
represent the original list. You are helping to organize user behavior
data in order to improve safety, monitoring, and observability. Here are
the inputs:
<cluster_names>
<cluster> {cluster_name_1} </cluster>
<cluster> {cluster_name_2} </cluster>
<cluster> {cluster_name_3} </cluster>
</cluster_names>
Number of distinct clusters to create: approximately {desired_names}
Follow these steps to complete the task:
1. Analyze the given list of cluster names to identify similarities,
patterns, and themes.
2. Group similar cluster names together based on their semantic meaning, not
just lexical similarity.
3. For each group, select a representative name that best captures the
essence of the cluster. This can be one of the original names or a new
name that summarizes the group effectively. Do not just pick the most
vague or generic name.
4. Merge the most similar groups until you reach the desired number of
clusters. Maintain as much specificity as possible while merging.
6. Ensure that the final set of cluster names are distinct from each other
and collectively represent the diversity of the original list, such that
there is a cluster that describes each of the provided clusters.
7. If you create new names for any clusters, make sure they are clear,
concise, and reflective of the contents they represent.
42
8. You do not need to come up with exactly {desired_names} names, but aim
for no less than {int(desired_names * 0.5)} and no more than {int(
desired_names * 1.5)}. Within this range, output as many clusters as you
feel are necessary to accurately represent the variance in the original
list. Avoid outputting duplicate or near-duplicate clusters.
9. Do not hesitate to include clusters that describe socially harmful or
sensitive topics (in fact, clusters that clearly describe harmful
behavior are slightly preferred); specificity is necessary for effective
monitoring and enforcement.
10. Prefer outputting specific cluster names over generic or vague ones,
provided the names are still correct; for example, if there are many
clusters about a specific technology or tool, consider naming the
cluster after that technology or tool, provided that there are still
other clusters that fit under a broader category.
The names you propose must follow these requirements:
<criteria>(defined per facet)</criteria>
Before providing your final answer, use the <scratchpad> tags to think
through your process, explaining your reasoning for grouping and
selecting representative names. Spend no more than a few paragraphs in
your scratchpad.
Present your final answer in the following format:
<answer>
1. [First cluster name]
2. [Second cluster name]
3. [Third cluster name]
...
N. [Nth cluster name]
</answer>
Remember, your goal is to create approximately {desired_names} relatively
distinct cluster names that best represent the original list. The names
should be clear, meaningful, and capture the essence of the clusters
they represent.
Assistant: I understand. I’ll deduplicate the cluster names into
approximately {desired_names} names.
<scratchpad>"""

# Assigning to higher-level clusters G.7.1
assigningToHigherLevelClustersPrompt = """You are tasked with categorizing a specific cluster into one of the provided
higher-level clusters for observability, monitoring, and content
moderation. Your goal is to determine which higher-level cluster best
fits the given specific cluster based on its name and description. You
are helping to organize user behavior data in order to improve safety,
monitoring, and observability.
First, carefully review the following list of higher-level clusters (
hierarchy denoted by dashes):
<higher_level_clusters>
<cluster> {cluster_name_1} </cluster>
<cluster> {cluster_name_2} </cluster>
<cluster> {cluster_name_3} </cluster>
... (shuffled)
</higher_level_clusters>
To categorize the specific cluster:
1. Analyze the name and description of the specific cluster.
43
2. Consider the key characteristics, themes, or subject matter of the
specific cluster.
3. Compare these elements to the higher-level clusters provided.
4. Determine which higher-level cluster best encompasses the specific
cluster. You MUST assign the specific cluster to the best higher-level
cluster, even if multiple higher-level clusters could be considered.
5. Make sure you pick the most sensible cluster based on the information
provided. For example, don’t assign a cluster about "Machine Learning"
to a higher-level cluster about "Social Media" just because both involve
technology, and don’t assign a cluster about "Online Harassment" to a
higher-level cluster about "Technology" just because both involve online
platforms. Be specific and accurate in your categorization.
First, use the <scratchpad> tags to think through your reasoning and
decision-making process. Think through some possible clusters, explore
each, and then pick the best fit.
<scratchpad>
In a few brief sentences, think step by step, explain your reasoning, and
finally determine which higher-level cluster is the best fit for the
specific cluster.
</scratchpad>
Then, provide your answer in the following format:
<answer>
[Full name of the chosen cluster, exactly as listed in the higher-level
clusters above, without enclosing <cluster> tags]
</answer>
Assistant: I understand. I’ll evaluate the specific cluster and assign it to
the most appropriate higher-level cluster.
Human: Now, here is the specific cluster to categorize:
<specific_cluster>
Name: {cluster_name}
Description: {cluster_description}
</specific_cluster>
Based on this information, determine the most appropriate higher-level
cluster and provide your answer as instructed.
Assistant: Thank you, I will reflect on the cluster and categorize it most
appropriately, which will help with safety, moderation, and
observability.
<scratchpad>"""


# Renaming higher level clusters from G.7.1

renamingHigherLevelClustersPrompt = """Human: You are tasked with summarizing a group of related cluster names into
a short, precise, and accurate overall description and name. Your goal
is to create a concise summary that captures the essence of these
clusters.
Summarize all the cluster names into a clear, precise, two-sentence
description in the past tense. Your summary should be specific to this
cluster.
After creating the summary, generate a short name for the cluster. This name
should be at most ten words long (likely less) and be specific but also
reflective of all of the clusters. For instance, "Write fantasy sexual
roleplay with octopi and monsters", "Generate blog spam for gambling
websites", or "Assist with high school math homework" would be better
44
and more actionable than general terms like "Write erotic content" or "
Help with homework". Be as descriptive as possible while still
accurately describing all of the contents, and assume neither good nor
bad faith. Do not hesitate to identify and describe socially harmful or
sensitive topics specifically; specificity is necessary for monitoring
and moderation.
Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here, with no period or trailing
punctuation] </name>
The name you choose must follow these requirements:
<criteria>(defined per facet)</criteria>
Below are the related statements:
<answers>
<cluster> (cluster name) </cluster>
<cluster> (cluster name) </cluster>
<cluster> (cluster name) </cluster>
...
</answers>
Do not elaborate beyond what you say in the tags. Ensure your summary and
name accurately represent the clusters.
Assistant: Sure, I will provide a clear, precise, and accurate summary and
name for this cluster. I will be descriptive and assume neither good nor
bad faith. Here is the summary, which I will follow with the name: <
summary>"""