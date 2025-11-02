# Chroniques Oubliees Fantasy simulator

This is a simple combat simulator for the table-top role-playing game 
Chroniques Oubliees.

It generates probability of winning a combat between two characters or groups of
characters as well as the average damage dealt, round after round.

## Defining Characters

The `character.py` file can be used to create individual Characters or 
groups of Characters.

For instance:
```
from character import Character
zeke = Character('Zeke', 20, 35, 0, 5, '2d6+1', critical=2)
```

will create a Character named Zeke, with a defence of 20, 35 hit points, no reduction
to damage, a +5 bonus to hit, and 2d6+1of damage dealt. In addition, Zeke hits a 
critical on a 19 and 20 so he has 2 chances to deal a critical hit.

## Attack and Fight

Once you have defined two characters, you can simulate an attack from one of them on
the other. For instance:

```
damage_values, damage_probabilities, average_damage = zeke.attacks(skeleton)
```

This can then be plotted

## Defining Groups

You can then combine characters into a group by doing:

> heroes = Character('Heroes', members=[zeke, thorfinn, janny, griselda, emerald, kendrika])

where each element of `members` is itself a Character.