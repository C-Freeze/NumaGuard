LAYUP_ATTEMPT = 0
SHOT_ATTEMPT = 1

player_dict = {
    "Player 1" : 0,
    "Player 2" : 1,
    "Player 3" : 2,
    "Player 4" : 3,
    "No One" : -1,
}



def get_attmpt_num(text: str):
    if text == "Shot":
        return 1,0
    else:
        return 0,1
    
def get_shooter_vals(text: str):
    
    if text == "Player 1":
        return 1, 0, 0, 0
    elif text == "Player 2":
        return 0, 1, 0, 0
    elif text == "Player 3":
        return 0, 0, 1, 0
    elif text == "Player 4":
        return 0, 0, 0, 1
    else:
        return 0, 0, 0, -1
    
def get_passer_vals(text: str):
    
    if text == "Player 1":
        return 1, 0, 0, 0, 0
    elif text == "Player 2":
        return 0, 1, 0, 0, 0
    elif text == "Player 3":
        return 0, 0, 1, 0, 0
    elif text == "Player 4":
        return 0, 0, 0, 1, 0
    else:
        return 0, 0, 0, 0, 1