import json


market_label_list = [
    "young",
    "teenager",
    "adult",
    "old",
    "backpack",
    "bag",
    "handbag",
    "clothes",
    "down",
    "up",
    "hair",
    "hat",
    "gender",
    "upblack",
    "upwhite",
    "upred",
    "uppurple",
    "upyellow",
    "upgray",
    "upblue",
    "upgreen",
    "downblack",
    "downwhite",
    "downpink",
    "downpurple",
    "downyellow",
    "downgray",
    "downblue",
    "downgreen",
    "downbrown"
]


duke_label_list = [
    "backpack",
    "bag",
    "handbag",
    "boots",
    "gender",
    "hat",
    "shoes",
    "top",
    "upblack",
    "upwhite",
    "upred",
    "uppurple",
    "upgray",
    "upblue",
    "upgreen",
    "upbrown",
    "downblack",
    "downwhite",
    "downred",
    "downgray",
    "downblue",
    "downgreen",
    "downbrown"
]


market_attribute_dict = {
    "young":        ["age", [None, "young"]],
    "teenager":     ["age", [None, "teenager"]],
    "adult":        ["age", [None, "adult"]],
    "old":          ["age", [None, "old"]],
    "backpack":     ["carrying backpack", ["no", "yes"]],
    "bag":          ["carrying bag", ["no", "yes"]],
    "handbag":      ["carrying handbag", ["no", "yes"]],
    "clothes":      ["type of lower-body clothing", ["dress", "pants"]],
    "down":         ["length of lower-body clothing", ["long lower body clothing", "short"]],
    "up":           ["sleeve length", ["long sleeve", "short sleeve"]],
    "hair":         ["hair length", ["short hair", "long hair"]],
    "hat":          ["wearing hat", ["no", "yes"]],
    "gender":       ["gender", ["male", "female"]],
    "upblack":      ["color of upper-body clothing", [None, "black"]],
    "upwhite":      ["color of upper-body clothing", [None, "white"]],
    "upred":        ["color of upper-body clothing", [None, "red"]],
    "uppurple":     ["color of upper-body clothing", [None, "purple"]],
    "upyellow":     ["color of upper-body clothing", [None, "yellow"]],
    "upgray":       ["color of upper-body clothing", [None, "gray"]],
    "upblue":       ["color of upper-body clothing", [None, "blue"]],
    "upgreen":      ["color of upper-body clothing", [None, "green"]],
    "downblack":    ["color of lower-body clothing", [None, "black"]],
    "downwhite":    ["color of lower-body clothing", [None, "white"]],
    "downpink":     ["color of lower-body clothing", [None, "pink"]],
    "downpurple":   ["color of lower-body clothing", [None, "purple"]],
    "downyellow":   ["color of lower-body clothing", [None, "yellow"]],
    "downgray":     ["color of lower-body clothing", [None, "gray"]],
    "downblue":     ["color of lower-body clothing", [None, "blue"]],
    "downgreen":    ["color of lower-body clothing", [None, "green"]],
    "downbrown":    ["color of lower-body clothing", [None, "brown"]],
}


duke_attribute_dict = {
    "backpack":     ["carrying backpack", ["no", "yes"]],
    "bag":          ["carrying bag", ["no", "yes"]],
    "handbag":      ["carrying handbag", ["no", "yes"]],
    "boots":        ["wearing boots", ["no", "yes"]],
    "gender":       ["gender", ["male", "female"]],
    "hat":          ["wearing hat", ["no", "yes"]],
    "shoes":        ["color of shoes", ["dark", "light"]],
    "top":          ["length of upper-body clothing", ["short upper body clothing", "long"]],
    "upblack":      ["color of upper-body clothing", [None, "black"]],
    "upwhite":      ["color of upper-body clothing", [None, "white"]],
    "upred":        ["color of upper-body clothing", [None, "red"]],
    "uppurple":     ["color of upper-body clothing", [None, "purple"]],
    "upgray":       ["color of upper-body clothing", [None, "gray"]],
    "upblue":       ["color of upper-body clothing", [None, "blue"]],
    "upgreen":      ["color of upper-body clothing", [None, "green"]],
    "upbrown":      ["color of upper-body clothing", [None, "brown"]],
    "downblack":    ["color of lower-body clothing", [None, "black"]],
    "downwhite":    ["color of lower-body clothing", [None, "white"]],
    "downred":      ["color of lower-body clothing", [None, "red"]],
    "downgray":     ["color of lower-body clothing", [None, "gray"]],
    "downblue":     ["color of lower-body clothing", [None, "blue"]],
    "downgreen":    ["color of lower-body clothing", [None, "green"]],
    "downbrown":    ["color of lower-body clothing", [None, "brown"]],
}


with open('./label.json', 'w') as f:
    label_list_dict = {
        'market': market_label_list,
        'duke': duke_label_list,
    }
    jsObj = json.dumps(label_list_dict, indent=4)
    f.write(jsObj)


with open('./attribute.json', 'w') as f:
    attribute_list_dict = {
        'market': market_attribute_dict,
        'duke': duke_attribute_dict,
    }
    jsObj = json.dumps(attribute_list_dict, indent=4)
    f.write(jsObj)