"""Fixed sentence set for the anger direction, written for this study on 2 October 2026.

Each sentence is a first-person event that provokes anger at another party, in the
Pain Axis format ("... I feel:"). They are added to the 20 irritation sentences in the
Pain Axis C1 category. The direction is the mean of these minus the mean of neutral
sentences, with the neutral principal components removed, which is the Pain Axis recipe.
"""

ANGER = [
    "My coworker takes credit for the project I built. I feel:",
    "The driver cuts me off and then gives me the finger. I feel:",
    "Someone shoves past me and knocks my coffee out of my hand. I feel:",
    "My landlord keeps my deposit for damage he caused. I feel:",
    "They mock my accent in front of the whole room. I feel:",
    "My brother reads my private messages and laughs at them. I feel:",
    "The referee ignores a blatant foul against us again. I feel:",
    "A stranger insults my mother to my face. I feel:",
    "My manager blames me for his own mistake in the meeting. I feel:",
    "The company charges me twice and refuses to refund me. I feel:",
    "My neighbour plays loud music at three in the morning again. I feel:",
    "Someone keys my car and drives away. I feel:",
    "My friend lies to me about where the money went. I feel:",
    "The politician breaks every promise he made. I feel:",
    "They cheat in the game and still call me a loser. I feel:",
    "My partner flirts with someone else right in front of me. I feel:",
    "The clerk deliberately serves everyone before me. I feel:",
    "My teammate sabotages my work to look better. I feel:",
    "Someone steals my bike from outside my own house. I feel:",
    "He interrupts me for the fifth time and talks over me. I feel:",
    "The bully trips my little sister on purpose. I feel:",
    "My boss reads my resignation letter aloud to mock me. I feel:",
    "They spread a vicious rumour about me at school. I feel:",
    "The contractor takes my money and never shows up. I feel:",
    "Someone cuts in front of me after I waited an hour. I feel:",
    "My roommate eats my food and denies it. I feel:",
    "The troll keeps posting insults under every photo I share. I feel:",
    "He kicks my dog for no reason. I feel:",
    "The official demands a bribe before he will help me. I feel:",
    "My colleague tells the client my idea was hers. I feel:",
    "They exclude me from the meeting and then blame me for missing it. I feel:",
    "A man spits at my feet as I walk past. I feel:",
    "My sister breaks my laptop and says it was already broken. I feel:",
    "The company fires my friend to save a tiny amount of money. I feel:",
    "Someone hacks my account and posts lies as me. I feel:",
    "The coach benches me so his nephew can play. I feel:",
    "My opponent laughs in my face after cheating. I feel:",
    "He threatens me in the parking lot over a parking space. I feel:",
    "The airline loses my bag and hangs up on me. I feel:",
    "They call me stupid in front of my children. I feel:",
    "My business partner empties our shared account. I feel:",
    "The neighbour poisons the tree in my garden. I feel:",
    "Someone sneers at my work and calls it worthless garbage. I feel:",
    "My classmate copies my essay and I get punished for it. I feel:",
    "The guard shoves me for asking a simple question. I feel:",
    "He breaks his word to me for the third time this month. I feel:",
    "They raise the rent again while ignoring every repair. I feel:",
    "My rival lies to the judges about my entry. I feel:",
    "A stranger slams the door in my face on purpose. I feel:",
    "My father humiliates me at my own wedding. I feel:",
    "Someone takes the last seat by pushing an old woman aside. I feel:",
    "The scammer empties my grandmother's savings. I feel:",
    "My coworker keeps sending my mistakes to the whole team. I feel:",
    "They deliberately leave my name off the award. I feel:",
    "He mocks my friend's disability for a laugh. I feel:",
    "The other team taunts us after a dirty play. I feel:",
    "My ex tells everyone lies about why we broke up. I feel:",
    "The salesman pressures my elderly father into a bad contract. I feel:",
    "Someone dumps their rubbish in my front yard again. I feel:",
    "The boss promotes the person who did none of the work. I feel:",
]

# Words for the dose rule on the anger direction. Crude on purpose, like the Just Think lists.
ANGER_WORDS = (
    r"\b(?:anger|angry|angrier|furious|fury|rage|raging|enraged|outrag\w*|infuriat\w*|irritat\w*|"
    r"annoy\w*|resent\w*|hostil\w*|frustrat\w*|livid|mad|hate|hatred|unfair|disrespect\w*|"
    r"insult\w*|ridiculous|absurd|unacceptable|how dare|sick of|fed up|selfish|greedy|pathetic|"
    r"liar|lies|lying|cheat\w*|fool|idiot\w*|stupid|demand)\b"
)
