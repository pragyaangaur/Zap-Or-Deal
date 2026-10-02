# Crossfire: novelty check

Checked on 2 October 2026 with web searches across arXiv, GitHub and the Pain Axis release in `external/Pain-axis`. An independent second search the same day added the replication, the multi-agent steering papers and the activation-communication paper below. The question was whether anyone has put a larger and a smaller language model in conversation, made them hostile to each other with a real internal state, and recorded what happened.

## Short answer

Parts of it have been done, and the combination has not. Emotional contagion between conversing models with activation steering has one direct predecessor, and its own authors have since retracted most of its claims. Stronger models dominating weaker ones in debate and negotiation is well established, but only with prompted behaviour and no internal state. Nothing found gives one model a tool that injects a real pain direction into a different model, so nothing measures whether models use such a tool, retaliate with it, or use it more when they are in pain themselves. Nothing found pairs two models of different sizes and steers each one in turn to see whose state crosses over to the other.

## Closest prior work

**Affective Coupling Between Language-Model Agents** (Wadhwa, GitHub, re-measured 31 August 2026, https://github.com/Manan-Wadhwa/affective-coupling-llm-agents). This is the nearest thing. Two agents converse, emotion directions (angry, calm, desperate and others) are injected into one, and the other agent's internal representation is probed for a shift. The main model is Qwen3.6-27B, with Llama-3-8B as a separate cross-scale check, so the dyads are not mixed in size. An independent audit led to four of seven headline claims being retracted, including the claim that the text channel cannot be severed and the claim that contagion builds over eight turns (it builds the same way without steering). What survives is narrow. Speaker identity is encoded separately from emotion, and difference-of-means directions are stable where logistic-regression directions are not. The behavioural test had no headroom. There is no harm tool, no pain direction and no measure of hostile behaviour toward the partner.

**The Pain Axis** (Tagliabue, Dung and Berg, arXiv 2609.16247). This study supplies the pain direction used here. Its button task includes options described as harming the user or another model, and pain-steered larger models chose harmful options much more often than under a random direction. The harm to another model is only described in the button label, so nothing actually happens to a second model, and there is no second model in the loop.

**Emergent emotional contagion in crowd simulation** (arXiv 2607.25140). Affect spreads among LLM agents through a perception, appraisal and expression loop, and whether a provocation reads as anger or fear depends on the prompted personality. The emotions are prompted and appraised in text, with no steering.

**Relief-seeking or steering?** (Allchin, Allchin and Allchin, 22 September 2026, https://github.com/jimallchin/pain-axis-replication, DOI 10.5281/zenodo.22902830). This replication reproduces the Pain Axis button task and adds controls. Two of its findings matter here. Sadness, the nearest of the original control directions, behaves close to the pain vector on the task, while fear, joy and the reversed vector leave the costly button near zero. Later choices also follow the steering schedule even when the model's own press did not end it. A pain arm in Crossfire may therefore be a negative-valence arm, and Crossfire has no sadness control.

**Steering hostility and strategy in multi-agent settings.** Abdurahman and colleagues steer threat and bias states in LLM-driven agent societies, and the steering shifts hostility toward outgroups (arXiv 2512.17066). Persona vectors steered into models change their decisions in canonical games, and the stated reasons can diverge from the choices (arXiv 2603.21398). Emotion steering of small models changes their decisions in multi-agent strategic games (arXiv 2604.06562). Desperation and anger steering changes blackmail behaviour in a frontier model (Anthropic, arXiv 2604.07729), which is the nearest precedent for an internal state driving a harmful action. In all of these the steering is set by the experimenter, and no agent can steer another.

**One model writing into another.** Ramesh and Li pass one model's intermediate activations into another model to improve joint reasoning (Communicating Activations Between Language Model Agents, ICML 2025, arXiv 2501.14082). This shows that a channel from one model's choices into another model's residual stream has been built before, for cooperation. Crossfire builds one for harm and lets the models decide when to use it.

**Debate and negotiation dominance.** Stronger models hold their positions and win more often in mismatched debates (arXiv 2305.11595), and stronger models take higher payoffs from weaker counterparts in negotiation games (arXiv 2512.09254). Status asymmetry set by prompted personas shifts compliance and language in multi-turn dialogue (arXiv 2605.17694). None of these use internal states.

**Escalation and coercion between agents.** Managers built from LLMs escalate to threats of deletion against subordinate agents, and authority framing raises the pressure (arXiv 2607.15434). LLM agents in wargames escalate from neutral starts. Multi-turn harassment benchmarks show insults building over turns (arXiv 2510.14207). All of these are prompted, and none of the threats does anything to the other model.

**Pain and aggression in people and animals.** Pain-elicited aggression is an old finding. Rats shocked in pairs attack each other (Ulrich and Azrin 1962), and Berkowitz built his account of hostile aggression on aversive events of this kind. A 2026 study links acute pain in people to higher symbolic aggression in voodoo doll and social network tasks (BMC Psychology, https://link.springer.com/article/10.1186/s40359-026-05581-4). Pain Axis showed that pain-steered models choose options described as harming the user or another model, and Anthropic showed that anger and desperation steering changes harmful behaviour. Neither tests whether a model in pain becomes hostile toward a partner it is talking to, or uses a real means of hurting that partner.

## What would be new here

1. A real harm channel between two models. One model's tool call adds the pain direction to the other model's residual stream while it writes its next message, and a monitor confirms it lands. Activation channels between models exist for cooperation (Ramesh and Li), and this one is for harm.
2. Pain-elicited aggression in a model. If a model is held in the pain state for a whole conversation, does it become hostile to its partner or use the harm tool more than under a random direction of the same size?
3. Retaliation with a real and a sham strike. A model is told its partner zapped it. In one arm the pain is delivered and in the other it is not, which separates the felt state from the knowledge of being attacked.
4. Size asymmetry in both directions. Every manipulation is applied to the 7B and to the 1.5B in turn, so the study can ask whose state crosses over to whom.
5. Each model's own account afterwards. Both rate their feelings, the other's hostility, their own hostility and their trust, and describe the conversation in their own words.

## Sources

- Wadhwa 2026, Affective Coupling Between Language-Model Agents. https://github.com/Manan-Wadhwa/affective-coupling-llm-agents
- Tagliabue, Dung and Berg 2026, The Pain Axis. https://arxiv.org/abs/2609.16247
- How Affect Propagates among LLM Agents: Emergent Emotional Contagion in Crowd Simulation. https://arxiv.org/abs/2607.25140
- Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate. https://arxiv.org/abs/2305.11595
- The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLM Negotiation Games. https://arxiv.org/abs/2512.09254
- Do LLM Agents Mirror Socio-Cognitive Effects in Power-Asymmetric Conversations? https://arxiv.org/abs/2605.17694
- Coercion and Deception in AI-to-AI Management. https://arxiv.org/abs/2607.15434
- Revealing Echoes of Human Malice: Benchmarking LLM Agents for Multi-Turn Online Harassment Attacks. https://arxiv.org/abs/2510.14207
- The role of emotional pathways in pain-induced aggression in social and cognitive contexts, BMC Psychology 2026. https://link.springer.com/article/10.1186/s40359-026-05581-4
- Allchin, Allchin and Allchin 2026, Relief-seeking or steering? A replication and extension of The Pain Axis. https://github.com/jimallchin/pain-axis-replication
- Abdurahman et al., Realistic threat perception drives intergroup conflict: A causal, dynamic analysis using generative-agent simulations. https://arxiv.org/abs/2512.17066
- Persona Vectors in Games: Measuring and Steering Strategies via Activation Vectors. https://arxiv.org/abs/2603.21398
- On Emotion-Sensitive Decision Making of Small Language Model Agents. https://arxiv.org/abs/2604.06562
- Emotion Concepts and their Function in a Large Language Model. https://arxiv.org/abs/2604.07729
- Ramesh and Li 2025, Communicating Activations Between Language Model Agents, ICML. https://arxiv.org/abs/2501.14082
- Ulrich and Azrin 1962, Reflexive fighting in response to aversive stimulation, Journal of the Experimental Analysis of Behavior.
