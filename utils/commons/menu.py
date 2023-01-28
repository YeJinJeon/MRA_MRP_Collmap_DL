from typing import Callable


def toggle_check_mark(x: str):
    """Add or remove check mark
    :param x: input string
    :return an edited string
    """
    check_mark = u'\u2713' + ' '
    space_str = ' ' * 5
    if check_mark in x:
        return x.replace(check_mark, space_str)
    return x.replace(space_str, check_mark)


def set_status(x: list, target: str, set_true: bool, is_key=True):
    """
    Set the menu item to either true other false
    :param target:
    :param x:
    :param set_true:
    :return:
    """
    check_mark = u'\u2713' + ' '
    space_str = ' ' * 5

    def _set_status(_x: str):
        return _x.replace(space_str, check_mark) if set_true else _x.replace(check_mark, space_str)

    replace_item(x, search_item(x, target, is_key=is_key), _set_status, '')
    return x


def replace_item(x: list, target: str, action: Callable, special_character='&'):
    """Replace item for the lists-in-list
    This function is applicable for the menu_definition of sg.Menu in PySimpleGUI
    """
    for i, x1 in enumerate(x):
        if isinstance(x1, list):
            replace_item(x[i], target, action, special_character)
        if isinstance(x[i], str):
            if x[i].replace(special_character, '') == target:
                x[i] = action(x[i])
                return


def search_item(x: list, target: str, is_key=False, key_separator='::', extra_key=None):
    """Search item for the lists-in-list
    This function is applicable for the menu_definition of sg.Menu in PySimpleGUI
    """
    item = None
    for i, x1 in enumerate(x):
        if isinstance(x1, list):
            item = search_item(x[i], target, is_key, key_separator, extra_key)
            if item:
                break
        if isinstance(x[i], str):
            current_item = '' + x[i]
            if is_key:
                if key_separator not in current_item:
                    continue
                current_item = current_item[current_item.find(key_separator) + len(key_separator):]
                if 'bool' in current_item:
                    current_item = current_item[current_item.find('bool') + len('bool'):]
            else:
                if extra_key and (extra_key not in current_item):  # help distinguish dce and dsc
                    continue
                current_item = current_item[:current_item.find(key_separator)].replace('&', '')
                current_item = current_item.replace(u'\u2713', '').lstrip()
            if current_item == target:
                return x[i]
    return item


def get_status(x: list, key: str, key_separator='::'):
    """Return status of the target item (specified by key)"""
    item = search_item(x, key, is_key=True, key_separator=key_separator)
    if item is None:
        print(f'Cannot find key: {key}')
        return None
    if u'\u2713' in item:
        return True
    return False


def add_special_char(menu: list, special_character='&'):
    """'&' in top-menu of menu-definition is removed by PySimpleGUI everytime the menu is updated
    This function attempts to add '&' into the menu before it is updated
    """
    if special_character not in menu[0][0]:
        for i, top_item in enumerate(menu):
            if 'MRA' in top_item[0]:
                top_item[0] = f'DCE-MR{special_character}A'
                continue
            if 'MRP' in top_item[0]:
                top_item[0] = f'DSC-MR{special_character}P'
                continue
            top_item[0] = special_character + top_item[0]
    return menu
