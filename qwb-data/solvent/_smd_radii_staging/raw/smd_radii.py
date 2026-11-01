def smd_radii(alpha):
    '''
    eq. (16)
    use smd radii if defined
    use Bondi radii if defined
    use 2.0 otherwise
    '''
    radii_table = radii.VDW.copy() * radii.BOHR
    radii_table[1] = 1.20
    radii_table[6] = 1.85
    radii_table[7] = 1.89
    if alpha >= 0.43:
        r = 1.52
    else:
        r = 1.52 + 1.8 * (0.43 - alpha)
    radii_table[8] = r
    radii_table[9] = 1.73
    radii_table[14] = 2.47
    radii_table[15] = 2.12
    radii_table[16] = 2.49
    radii_table[17] = 2.38
    #radii_table[35] = 3.06 # original SMD
    # following value from SMD18
    # https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.201803652
    radii_table[35] = 2.60
    radii_table[53] = 2.74
    return radii_table/radii.BOHR
