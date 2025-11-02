import numpy as np
import re


class Character:
    def __init__(self, name,
                 defence=None, hitpoints=None, reduc_damage=0,
                 attack=None, damage=None, critical=1, aoe=1,
                 members=None):
        """
        Parameters
        ----------
        name: string
            a name for the Character object
        members: None or list of Character objects
            if provided, the Character is a group of Character objects, otherwise it is a single Character object
        defence: None or int
            defence score
        hitpoints: None or int
            available hit points
        reduc_damage: int
            damage reduction
        attack: None or int
            bonus to hit with the attack
        damage: None or string
            damage to roll for the weapon as a string (e.g. '1d8+2')
        critical: Int
            number of values on d20 that are a critical hit (e.g. 2 if critical is 19-20)
        aoe: Int
            number of enemies that gets hit and damaged by the attack
        """
        self.name = name
        # Figure out if this is an individual Character of a Group of Characters
        if members:
            self.members = members if len(members) > 1 else None
        else:
            self.members = None
        if self.members:
            print('This is a Group of Character objects')
            self.defence = int(round(np.mean([m.defence for m in members])))
            self.hitpoints = np.sum([m.hitpoints for m in members])
            self.reduc_damage = int(round(np.mean([m.reduc_damage for m in members])))
            print(f'... using mean Defence score {self.defence} and total hit points {self.hitpoints}')
            # if a group of Character, the attack, damage, and critical are defined for each member
            self.attack = None
            self.damage = None
            self.critical = None
            self.aoe = None
        else:
            print('This is a Character object')
            self.defence = defence
            self.hitpoints = hitpoints
            self.reduc_damage = reduc_damage
            self.attack = attack
            self.damage = damage
            self.critical = critical
            self.aoe = aoe

        # todo: add a property that tells how many enemies an attack can hit (AoE attacks)

    @property
    def is_group(self):
        return self.members is not None


    def attacks(self, target_character, verbose=False):

        # Is the attacking Character a Group or an Individual?
        if self.is_group:
            # Combine all attacking members' attacks against the enemy Character or Group of Character
            group_dmg_p = np.array([1.0])
            for member in self.members:
                v, p, _ = member.attacks(target_character, verbose=verbose)
                group_dmg_p = np.convolve(group_dmg_p, p)
            group_dmg_v = np.arange(0, len(group_dmg_p))
            # Average damage for the whole group
            avg_group_dmg = np.sum(group_dmg_v * group_dmg_p)

            return group_dmg_v, group_dmg_p, avg_group_dmg

        else:
            # compute proba to hit (assuming standard attack)
            to_hit = target_character.defence - self.attack
            # accounting for critical miss and critical hit
            hit_proba = self.critical / 20 if to_hit > 20 \
                else 19. / 20 if to_hit < 1 \
                else (20 - to_hit) / 20.
            normal_hit_p = hit_proba - self.critical/20
            crit_hit_p = self.critical/20
            miss_p = 1 - hit_proba
            # compute proba for normal damage
            dice, modifier = parse_dice_expression(self.damage)
            dmg_v, dmg_p = roll_dice_proba(dice, modifier)

            # compute proba for critical damage
            dmg_v_crit = 2 * dmg_v
            dmg_p_crit = dmg_p * 1

            # combine proba to hit with dmg proba
            dmg_p *= normal_hit_p
            dmg_p_crit *= crit_hit_p

            # combine normal and critical distributions
            all_dmg_v = np.arange(0, np.max(dmg_v_crit) + 1)
            all_dmg_p = np.zeros_like(all_dmg_v, dtype=float)
            # normal hits
            for v, p in zip(dmg_v, dmg_p):
                all_dmg_p[v] += p
            # critical hits
            for v, p in zip(dmg_v_crit, dmg_p_crit):
                all_dmg_p[v] += p
            # misses
            all_dmg_p[0] = miss_p

            # reduce all damage by damage reduction of target
            all_dmg_v -= target_character.reduc_damage
            # combine all dmg <= 0
            if np.min(all_dmg_v) < 0:
                dmg_p_1 = np.sum(all_dmg_p[all_dmg_v <= 0])  # sum proba that dmg <= 0
                new_dmg_p = all_dmg_p[all_dmg_v > 0]  # remove all dmg <= 0
                new_dmg_v = all_dmg_v[all_dmg_v > 0]  # remove all dmg <= 0
                all_dmg_v = np.append(new_dmg_v, 0)  # add dmg=0 with correct proba
                all_dmg_p = np.append(new_dmg_p, dmg_p_1)  # add dmg=0 with correct proba
                all_dmg_p = all_dmg_p[np.argsort(all_dmg_v)]
                all_dmg_v = np.sort(all_dmg_v)

            # account for AoE attacks
            if target_character.is_group and self.aoe > 1:
                all_dmg_v *= min(self.aoe, len(target_character.members))

            # compute average dmg
            dmg_avg = np.sum(all_dmg_v * all_dmg_p)

            if verbose:
                print(f'{self.name} attacks {target_character.name}')
                if target_character.is_group and self.aoe > 1:
                    print('Target is a group and attack has AoE')
                print('Proba distribution of damage:')
                print(all_dmg_v)
                print(all_dmg_p)
                print('Average damage:')
                print(dmg_avg)

            return all_dmg_v, all_dmg_p, dmg_avg

    def fights(self, target_character, rounds=10, verbose=False):
        # Get the proba distri for one attack
        v1, p1, avg1 = self.attacks(target_character, verbose=verbose)
        # Use that as the starting point
        p = np.copy(p1)
        # Add each round
        dmg_p_list, dmg_v_list, dmg_avg_list = [p1], [v1], [avg1]
        kill_p_list = [np.sum(p1[v1 >= target_character.hitpoints])]

        for _ in range(rounds-1):
            # proba distri after that new round
            p = np.convolve(p, p1)
            v = np.arange(len(p))
            # store that proba distri
            dmg_p_list.append(p)
            dmg_v_list.append(v)
            # compute average damage
            dmg_avg = np.sum(v * p)
            dmg_avg_list.append(dmg_avg)
            # compute proba to kill
            kill_p = np.sum(p[v >= target_character.hitpoints])
            kill_p_list.append(kill_p)

        return dmg_p_list, dmg_v_list, dmg_avg_list, kill_p_list



def parse_dice_expression(expr):
    """
    Parse a dice expression such as '1d6+2d8+1d4+5' or '3d6-1'
    and return a tuple: (list_of_dice_sides, flat_modifier)

    Example
    -------
        '1d6+2d8+1d4+5' -> ([6, 8, 8, 4], 5)
        '1d12'          -> ([12], 0)
        '3d6-1'         -> ([6, 6, 6], -1)
    """
    dice = []
    modifier = 0

    # Remove all spaces for simplicity
    expr = expr.replace(" ", "")

    # Remove dice parts to find flat numeric modifiers (+5, -3, etc.)
    expr_no_dice = re.sub(r'[+-]?\d*d\d+', '', expr)
    for num in re.findall(r'[+-]?\d+', expr_no_dice):
        modifier += int(num)

    # Match dice terms like 2d8, 1d6, 3d4, etc.
    for count, sides in re.findall(r'([+-]?\d*)d(\d+)', expr):
        count = int(count.lstrip('+') or 1)
        sides = int(sides)
        dice.extend([sides] * count)

    return dice, modifier


def roll_dice_proba(nsides, offset):
    """
    Compute the probability distribution for a dice roll.

    Parameters
    ----------
    nsides : list of int
        List of the number of sides for each die (e.g., [6, 8] for 1d6+1d8).
    offset : int, optional
        Constant offset (bonus or penalty) added to the roll.

    Returns
    -------
    values : ndarray
        Array of possible summed values (with offset applied).
    proba : ndarray
        Corresponding probabilities for each possible outcome.

    Examples
    -------
    # To simulate a dice roll of 1d6 + 1d8:
    >>> values, proba = roll_dice_proba([6, 8], 0)
    """

    # Proba distribution for the first die
    proba = np.ones(nsides[0]) / nsides[0]

    # If only one die in the roll
    if len(nsides) == 1:
        # Compute the values, including the offset
        values = np.arange(1+offset, nsides[0]+1+offset)
        return values, proba

    # If more than one die in the roll
    else:
        # Combine each new die in terms of proba
        for i in range(1, len(nsides)):
            p = np.ones(nsides[i]) / nsides[i]
            proba = np.convolve(proba, p)
        # Now compute the values, including the offset
        values = np.arange(len(nsides)+offset, np.sum(nsides)+1+offset)
        return values, proba

