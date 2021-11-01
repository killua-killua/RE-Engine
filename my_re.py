import os
import sys

from typing import List, Tuple, Set, final


# Grammar tree
class GTNode():
    uid:int = 0

    def __init__(self):
        self.children = []
        self.min_times:int = 1
        self.max_times:int = 1
        self.uid = GTNode.uid
        GTNode.uid += 1
    
    def add_child(self, node):
        self.children.append(node)
    
    def set_times(self, min:int, max:int):
        self.min_times = min
        self.max_times = max

    # tree dump helper
    def to_string(self):
        pass

    def get_repeat_desc(self):
        s = ''
        if self.min_times == 1 and self.max_times == 1:
            pass
        elif self.min_times == 1 and self.max_times == -1:
            s = '+'
        elif self.min_times == 0 and self.max_times == -1:
            s = '*'
        elif self.min_times == 0 and self.max_times == 1:
            s = '?'
        else:
            s = '{' + str(self.min_times) + ',' + str(self.max_times) + '}'
        return s

    def display(self, s:str) -> str:
        ret = s
        if s == ' ':
            ret = r'\\s'
        elif s == '\t':
            ret = r'\\t'
        elif s == '\r':
            ret = r'\\r'
        elif s == '\n':
            ret = r'\\n'
        elif s == '\\':
            ret = r'\\'
        elif s == '"':
            ret = r'\"'
        return ret

    def dump_dot(self):
        s = f'{self.uid} [label="{self.to_string()}"]'
        print(s)
        for child in self.children:
            s = f'{child.uid} [label="{child.to_string()}"]'
            print(s)
            s = f'{self.uid} -> {child.uid}'
            print(s)
        for child in self.children:
            child.dump_dot()

class OrNode(GTNode):
    def __init__(self):
        super().__init__()
    
    def to_string(self):
        return 'Or ' + self.get_repeat_desc()
        
class AndNode(GTNode):
    def __init__(self):
        super().__init__()
    
    def to_string(self):
        return 'And'

class CharNode(GTNode):
    def __init__(self, c):
        super().__init__()
        self.cha = c
    
    def to_string(self):
        return self.display(self.cha) + self.get_repeat_desc()

class CharSetNode(GTNode):
    def __init__(self):
        super().__init__()
        self.char_set:List[Tuple[str,str]] = []
        self.exclude = False

    def add_char_set(self, c1, c2):
        self.char_set.append((c1, c2))

    def to_string(self):
        s = '['
        if self.exclude:
            s += '^'
        for pair in self.char_set:
            if pair[0] == pair[1]:
                s += self.display(pair[0])
            else:
                s += self.display(pair[0]) + '-' + self.display(pair[1])
        s += ']' + self.get_repeat_desc()
        return s


# NFA state & transitions
class Transition():
    pass

class State():
    uid:int = 0
    def __init__(self):
        self.uid = State.uid
        State.uid += 1
        self.trans:List[Transition] = []

    def add_tran(self, tran:Transition, front=False):
        if front:
            self.trans.insert(0, tran)
        else:
            self.trans.append(tran)
    
    def get_label(self):
        return str(self.uid)

class Transition():    
    def __init__(self, to_state, char_set=[], exclude=False):
        # if it's a epsilon tran, call Transition(to_state)
        self.toState:State = to_state
        self.char_set:List[Tuple[str,str]] = char_set
        self.exclude = exclude
    
    def is_epsilon(self):
        return True if len(self.char_set) == 0 else False

    def match(self, c) -> bool:
        if self.is_epsilon():
            return False
        ret = False
        for pair in self.char_set:
            if pair[0] <= c and c <= pair[1]:
                ret = True
                break
        if self.exclude:
            ret = not ret
        return ret

    # graph dump helper
    def display(self, s:str) -> str:
        ret = s
        if s == ' ':
            ret = r'\\s'
        elif s == '\t':
            ret = r'\\t'
        elif s == '\r':
            ret = r'\\r'
        elif s == '\n':
            ret = r'\\n'
        elif s == '\\':
            ret = r'\\'
        elif s == '"':
            ret = r'\"'
        return ret
    
    def get_label(self):
        if self.is_epsilon():
            return 'eps'
        if len(self.char_set) == 1 and self.char_set[0][0] == self.char_set[0][1]:
            return self.display(self.char_set[0][0])

        s = '['
        if self.exclude:
            s += '^'
        for pair in self.char_set:
            if pair[0] == pair[1]:
                s += self.display(pair[0])
            else:
                s += self.display(pair[0]) + '-' + self.display(pair[1])
        s += ']'
        return s


# DFA state & transitions
class DTran():
    pass

class DState():
    uid:int = 0
    def __init__(self, states:Set[State], accept):
        self.uid = DState.uid
        DState.uid += 1
        self.dtrans:List[DTran] = []
        self.states:Set[State] = states
        self.accept = accept
    
    def add_dtran(self, c:str, dState):
        for dtran in self.dtrans:
            if dtran.dState is dState:
                dtran.char_set.add(c)
                return
        dtran = DTran(dState, set(c))
        self.dtrans.append(dtran)
    
    def get_label(self):
        return str(self.uid)

class DTran():
    def __init__(self, dState, char_set):
        self.dState:DState = dState
        self.char_set:Set[str] = char_set

    def match(self, c) -> bool:
        return c in self.char_set

    # graph dump helper
    def display(self, s:str) -> str:
        ret = s
        if s == ' ':
            ret = r'\\s'
        elif s == '\t':
            ret = r'\\t'
        elif s == '\r':
            ret = r'\\r'
        elif s == '\n':
            ret = r'\\n'
        elif s == '\\':
            ret = r'\\'
        elif s == '"':
            ret = r'\"'
        return ret
    
    def get_label(self):
        if len(self.char_set) == 1:
            s = [c for c in self.char_set]
            return self.display(s[0])
        if len(self.char_set) == 2:
            s = [c for c in self.char_set]
            return self.display(s[0]) + ',' + self.display(s[1])
        return '...'



# RE object
class RE():
    def __init__(self, re):
        self.re = re
        self.gt:GTNode = None
        self.nfa:List[State] = []
        self.dfa:List[DState] = []
        self.parse_opti()

    def parse(self) -> GTNode:
        def get_repeat_times(s) -> Tuple[int, int, int]:
            ''' parse quantifier: * + ? {n, m} '''
            min = 1; max = 1; n = 0
            if s[0] == '*':
                min = 0; max = -1; n = 1
            elif s[0] == '+':
                min = 1; max = -1; n = 1
            elif s[0] == '?':
                min = 0; max = 1; n = 1
            return min, max, n
        
        def get_charactor(s) -> Tuple[str, int]:
            ''' parse a charactor, including escape ones '''
            c = ''
            n = 1
            if s[0] == '\\':
                if 1 >= len(s):
                    return None, 0
                if s[1] in ['\\', '+', '*', '?', '-', '(', ')', '[', ']']:
                    c = s[1]
                elif s[1] == 's':
                    c = ' '
                elif s[1] == 't':
                    c = '\t'
                elif s[1] == 'r':
                    c = '\r'
                elif s[1] == 'n':
                    c = '\n'
                else:
                    print('unsupported escape charactor: ' + s[1])
                    return None, 0
                n = 2
            else:
                c = s[0]
                n = 1
            return c, n
        
        def get_char_set(s) -> Tuple[CharSetNode, int]:
            char_set_list = []
            exclude = False
            i = 1
            if i < len(s) and s[i] == '^':
                exclude = True
                i += 1
            while i < len(s):
                if s[i] == ']':
                    if len(char_set_list) == 0:
                        print('no valid charactor in bracket')
                        return None, 0
                    node = CharSetNode()
                    node.exclude = exclude
                    for pair in char_set_list:
                        # print(pair[0], pair[1])
                        node.add_char_set(pair[0], pair[1])
                    return node, i+1

                c, n = get_charactor(s[i:])
                if c is None:
                    return None, 0
                i += n
                if i >= len(s):
                    return None, 0
                c2 = c
                if s[i] == '-':
                    i += 1
                    if i >= len(s):
                        return None, 0
                    c2, n = get_charactor(s[i:])
                    if c2 is None:
                        return None, 0
                    i += n
                char_set_list.append((c, c2))
            # lack of termination ']', return None
            return None, 0

        def parse_or(s:str) -> Tuple[GTNode, int]:
            node_or:GTNode = OrNode()
            node_and:GTNode = AndNode()
            node_or.add_child(node_and)

            i = 1
            while i < len(s):
                c = s[i]
                # first handle meta charactor
                if c == '(':
                    node, n = parse_or(s[i:])
                    if node is None:
                        return None, 0
                    node_and.add_child(node)
                    i += n
                    if i < len(s):
                        min, max, n = get_repeat_times(s[i:])
                        node.set_times(min, max)
                        i += n
                elif c == ')':  # termination
                    return node_or, i+1
                elif c == '|':
                    node_and = AndNode()
                    node_or.add_child(node_and)
                    i += 1
                elif c == '[':
                    node, n = get_char_set(s[i:])
                    if node is None:
                        return None, 0
                    node_and.add_child(node)
                    i += n
                    if i < len(s):
                        min, max, n = get_repeat_times(s[i:])
                        node.set_times(min, max)
                        i += n
                # then handle normal charactor
                else:
                    cha, n = get_charactor(s[i:])
                    if cha is None:
                        return None, 0
                    node = CharNode(cha)
                    node_and.add_child(node)
                    i += n
                    if i < len(s):
                        min, max, n = get_repeat_times(s[i:])
                        node.set_times(min, max)
                        i += n

            # lack of termination ')', return None
            return None, 0
        
        s = '(' + self.re + ')'
        gt, n = parse_or(s)
        if gt and n == len(s):
            self.gt = gt
        else:
            print('re parse error')
    
    def parse_opti(self):

        def get_repeat_times(s) -> Tuple[int, int, int]:
            ''' parse quantifier: * + ? {n, m} '''
            min = 1; max = 1; n = 0
            if s[0] == '*':
                min = 0; max = -1; n = 1
            elif s[0] == '+':
                min = 1; max = -1; n = 1
            elif s[0] == '?':
                min = 0; max = 1; n = 1
            return min, max, n
        
        def get_charactor(s) -> Tuple[str, int]:
            ''' parse a charactor, including escape ones '''
            c = ''
            n = 1
            if s[0] == '\\':
                if 1 >= len(s):
                    return None, 0
                if s[1] in ['\\', '+', '*', '?', '-', '(', ')', '[', ']']:
                    c = s[1]
                elif s[1] == 's':
                    c = ' '
                elif s[1] == 't':
                    c = '\t'
                elif s[1] == 'r':
                    c = '\r'
                elif s[1] == 'n':
                    c = '\n'
                else:
                    print('unsupported escape charactor: ' + s[1])
                    return None, 0
                n = 2
            else:
                c = s[0]
                n = 1
            return c, n
        
        def get_char_set(s) -> Tuple[CharSetNode, int]:
            char_set_list = []
            exclude = False
            i = 1
            if i < len(s) and s[i] == '^':
                exclude = True
                i += 1
            while i < len(s):
                if s[i] == ']':
                    if len(char_set_list) == 0:
                        print('no valid charactor in bracket')
                        return None, 0
                    node = CharSetNode()
                    node.exclude = exclude
                    for pair in char_set_list:
                        # print(pair[0], pair[1])
                        node.add_char_set(pair[0], pair[1])
                    return node, i+1

                c, n = get_charactor(s[i:])
                if c is None:
                    return None, 0
                i += n
                if i >= len(s):
                    return None, 0
                c2 = c
                if s[i] == '-':
                    i += 1
                    if i >= len(s):
                        return None, 0
                    c2, n = get_charactor(s[i:])
                    if c2 is None:
                        return None, 0
                    i += n
                char_set_list.append((c, c2))
            # lack of termination ']', return None
            return None, 0

        ''' non-recursive version '''

        s = self.re
        root:GTNode = None
        node_or:GTNode = None
        node_and:GTNode = None
        stack:List[GTNode] = []

        node_or = OrNode()  # context value, need save-restore
        node_and = AndNode()  # context value need save-restore
        node_or.add_child(node_and)
        root = node_or

        i = 0  # not context value, no need to save-restore
        while i < len(s):
            # first handle meta charactor
            if s[i] == '(':
                # save context
                stack.append(node_or)
                stack.append(node_and)
                node_or = OrNode()
                node_and.add_child(node_or)
                node_and = AndNode()
                node_or.add_child(node_and)
                i += 1
            elif s[i] == ')':
                if len(stack) < 2:
                    root = None
                    break
                i += 1
                # check quantity modifier
                if i < len(s):
                    min, max, n = get_repeat_times(s[i:])
                    node_or.set_times(min, max)
                    i += n
                # restore context
                node_and = stack.pop()
                node_or = stack.pop()
            elif s[i] == '|':
                node_and = AndNode()
                node_or.add_child(node_and)
                i += 1
            elif s[i] == '[':
                node, n = get_char_set(s[i:])
                if node is None:
                    root = None
                    break
                node_and.add_child(node)
                i += n
                if i < len(s):
                    min, max, n = get_repeat_times(s[i:])
                    node.set_times(min, max)
                    i += n
            # then handle normal charactor
            else:
                cha, n = get_charactor(s[i:])
                if cha is None:
                    root = None
                    break
                node = CharNode(cha)
                node_and.add_child(node)
                i += n
                if i < len(s):
                    min, max, n = get_repeat_times(s[i:])
                    node.set_times(min, max)
                    i += n
        
        if root is None or len(stack) > 0:
            print('re parse error')
        else:
            self.gt = root
        # 最后生成的gt，有几个特点：Or下面一定是And；And下面不会是And；括号一定会产生一条Or-And分支，即便re分支是空的，如 (a()b)
        # 对于一些简单的re表达式或子表达式，比如：a  (())，存在优化空间。
        # 可按照如下规则进行剪枝：若Or/And下面只有一个结点，则用 child替换 Or/And（注意：Null child 也算一个 child）
        # 例子:   a、((a))、(())、int|[a-zA-Z][a-zA-Z0-9]*|[0-9]+ （关注最后一条分支）
        # 对一条分支的优化，可能会触发另一条分支的优化，比如：(())|((a))
        # 注意：尽管这里可以做一些小的优化，让gt更简单，从而生成的NFA也更简单，但后面会有NFA->DFA
        # 这个本身就是一个优化过程，即使gt不优化，有很多冗余分支，它也能把冗余的NFA信息去除掉，得到简洁的DFA


    def to_nfa(self):
        def handle_quantity(state_list:List[State], min_times:int, max_times:int):
            if len(state_list) == 0:
                return
            if min_times == 1 and max_times == 1:
                return

            # 如果要确保将来用NFA匹配字符串的策略是贪婪/非贪婪，则这里要确保插入的 epsilon tran 的位置
            # 实现贪婪策略时，有些 epsilon tran 应该放到 trans 的前面，有的应放到后面，原则是让本unit内部能消耗掉更多字符
            # 每个被量词标记的unit，可以有不同的贪婪策略（与？修饰符有关，这里没有加入这个机制）
            # 下面的代码按默认的贪婪策略来

            initial:State = state_list[0]
            final:State = state_list[-1]
            if min_times == 1 and max_times == -1:  # +
                r_initial:State = State()
                r_final:State = State()
                r_initial.add_tran(Transition(initial))
                final.add_tran(Transition(r_final))
                final.add_tran(Transition(initial), True)  # Put the new tran at front, to be greedy match
                state_list.insert(0, r_initial)
                state_list.append(r_final)
            elif min_times == 0 and max_times == -1:  # *
                r_initial:State = State()
                r_final:State = State()
                r_initial.add_tran(Transition(initial))
                final.add_tran(Transition(r_final))
                final.add_tran(Transition(initial), True)  # greedy
                r_initial.add_tran(Transition(r_final))  # not greedy
                state_list.insert(0, r_initial)
                state_list.append(r_final)
            elif min_times == 0 and max_times == 1:  # ?
                r_initial:State = State()
                r_final:State = State()
                r_initial.add_tran(Transition(initial))
                final.add_tran(Transition(r_final))
                r_initial.add_tran(Transition(r_final))  # not greedy
                state_list.insert(0, r_initial)
                state_list.append(r_final)
            else:
                pass

        def to_nfa_recur(gt:GTNode) -> List[State]:
            state_list:List[State] = []

            if isinstance(gt, OrNode):
                initial:State = State()
                final:State = State()
                state_list.append(initial)
                # if gt is not optimized, child must be And. Otherwise, child may be Char/CharSet/And.
                for child in gt.children:
                    tmp_list = to_nfa_recur(child)
                    if len(tmp_list) == 0:
                        continue
                    # join tmp_list to state_list by its header & tail node
                    state_list.extend(tmp_list)
                    initial.add_tran(Transition(tmp_list[0]))
                    tmp_list[-1].add_tran(Transition(final))
                state_list.append(final)
                # Initial/final must be the header/tail of state_list. Then no need to care the order of State inside state_list.
                # The caller may join state_list to other part with its header and tail, parallely or serially.

                # check quantity
                handle_quantity(state_list, gt.min_times, gt.max_times)

            elif isinstance(gt, AndNode):
                for child in gt.children:
                    tmp_list = to_nfa_recur(child)
                    # if len(tmp_list == 0):
                    #     continue
                    if len(state_list) == 0:
                        state_list.extend(tmp_list)
                    else:
                        # 这里做了个优化：将And下的子结点前后合并，去掉不必要的epsilon分支
                        # 能按下面这样操作，是基于一个前提：tmp_list中的header，不会被tmp_list中的其它State指向（这是由对量词的处理保证的）
                        # 如果不能保证上述的前提，则需要实现一个merge(state_list, tmp_list)函数，确保已有的连接不会失效
                        state_list[-1].trans.extend(tmp_list[0].trans)
                        state_list.extend(tmp_list[1:])
                # And结点没有量词，不需要 handle_quantity

            elif isinstance(gt, CharNode):
                initial:State = State()
                final:State = State()
                cha = gt.cha
                initial.add_tran(Transition(final, [(cha, cha)]))
                state_list.extend([initial, final])
                # check quantity
                handle_quantity(state_list, gt.min_times, gt.max_times)

            elif isinstance(gt, CharSetNode):
                initial:State = State()
                final:State = State()
                char_set:List[Tuple[str, str]] = gt.char_set
                exclude = gt.exclude
                initial.add_tran(Transition(final, char_set, exclude))
                state_list.extend([initial, final])
                # check quantity
                handle_quantity(state_list, gt.min_times, gt.max_times)
                
            else:
                pass

            return state_list

        if self.gt is None:
            print('no gt')
            return

        self.nfa = to_nfa_recur(self.gt)

    def to_dfa(self):
        def get_epsilon_closure(states:Set[State]) -> Set[State]:
            def get_epsilon_closure_recur(state:State) -> Set[State]:
                ret_set:Set[State] = set()
                ret_set.add(state)
                for tran in state.trans:
                    if tran.is_epsilon():
                        tmp_set = get_epsilon_closure_recur(tran.toState)
                        ret_set.update(tmp_set)
                return ret_set

            ret_set:Set[State] = set()
            for state in states:
                tmp_set = get_epsilon_closure_recur(state)
                ret_set.update(tmp_set)
            return ret_set
        
        def get_char_closure(states:Set[State], c:str) -> Set[State]:
            ret_set:Set[State] = set()
            for state in states:
                for tran in state.trans:
                    if tran.match(c):
                        ret_set.add(tran.toState)
            return ret_set
        
        def is_set_equal(set1:Set[State], set2:Set[State]):
            return set1 == set2

        if len(self.nfa) == 0:
            print('no nfa')
            return
        
        # alphabet 是正则表达式可以识别的所有内容字符
        alphabet:Set[str] = [chr(i) for i in range(256)]

        dstates:List[DState] = []
        queue:List[DState] = []
        nfa_start = self.nfa[0]
        nfa_end = self.nfa[-1]

        states = get_epsilon_closure(set([nfa_start]))
        accept = nfa_end in states
        dstate = DState(states, accept)
        dstates.append(dstate)
        queue.append(dstate)

        while len(queue) > 0:
            src_dstate = queue.pop(0)  # BFS
            # debug purpose
            # s = ''
            # for state in src_dstate.states:
            #     s += state.get_label() + ','
            # print(s)
            for c in alphabet:
                states = get_char_closure(src_dstate.states, c)
                states = get_epsilon_closure(states)
                if len(states) > 0:
                    found = False
                    for tmp_dstate in dstates:
                        if is_set_equal(states, tmp_dstate.states):
                            found = True
                            src_dstate.add_dtran(c, tmp_dstate)
                            break
                    if not found:
                        accept = nfa_end in states
                        dstate = DState(states, accept)
                        dstates.append(dstate)
                        queue.append(dstate)
                        src_dstate.add_dtran(c, dstate)
        self.dfa = dstates
    

    def match_with_nfa(self, s) -> bool:
        if len(self.nfa) == 0:
            print('no nfa')
            return False
        
        def match_with_nfa_recur(s:str, i:int, nfa:List[State], state:State) -> bool:
            if i == len(s) and state == nfa[-1]:
                return True
            # 贪婪策略的实现，应该放到构造NFA的环节，这里只需要按tran的顺序依次尝试即可
            # 对于 re.match 类的应用，贪婪策略与否，不会影响最终的结果，但对于 re.search，则有很大影响
            for tran in state.trans:
                if tran.is_epsilon():
                    ret = match_with_nfa_recur(s, i, nfa, tran.toState)
                    if ret == True:
                        return True
                else:
                    if i < len(s) and tran.match(s[i]):
                        ret = match_with_nfa_recur(s, i+1, nfa, tran.toState)
                        if ret == True:
                            return True
            return False

        if len(s) == 0:
            return False    
        return match_with_nfa_recur(s, 0, self.nfa, self.nfa[0])
        
    def match_with_dfa(self, s) -> bool:
        if len(self.dfa) == 0:
            print('no dfa')
            return False
        
        dstate = self.dfa[0]
        i = 0
        while i < len(s):
            found = False
            for dtran in dstate.dtrans:
                if dtran.match(s[i]):
                    found = True
                    dstate = dtran.dState
                    break
            if not found:
                break
            else:
                i += 1
        if i < len(s):
            # 字符没有消耗完，就返回了，说明中间有个字符无法匹配任何分支
            return False
        else:
            return dstate.accept  # 检查是否停在了接受状态上

        
    def dump_nfa_dot(self):
        # dump graph
        print('digraph g {')
        print('rankdir=LR')
        for state in self.nfa:
            for tran in state.trans:
                s = f'{state.uid} -> {tran.toState.uid} [label="{tran.get_label()}"]'
                print(s)
        print('}')
    
    def dump_dfa_dot(self):
        # dump graph
        print('digraph g {')
        print('rankdir=LR')
        for dstate in self.dfa:
            for dtran in dstate.dtrans:
                s = f'{dstate.uid} -> {dtran.dState.uid} [label="{dtran.get_label()}"]'
                print(s)
            if dstate.accept:
                s = f'{dstate.uid} [shape="doublecircle"]'
                print(s)
        print('}')



# # test re parser ( re -> gt )
# if __name__ == '__main__':
#     re1 = "int|[a-zA-Z][a-zA-Z0-9]*|[0-9]+"
#     re2 = "int|[a-z](0|01)*a|[0-9]+"

#     re3 = "int|[a-z\\s-\\\\]*"
#     re4 = r"int|[a-z\s-\\]*"
#     re5 = '"\n[a\\s-\\t"]'
#     re6 = r"a?([a-z\s-\\]*)?b+"
#     re7 = "(int|1)"
#     re8 = "[^0-9]?"

#     re9 = "(int | 1)2)"
#     re10 = "int |(1"

#     re = RE(re1)
#     if re.gt:
#         re.gt.dump_dot()


# # test gt_to_nfa & match_with_nfa
# if __name__ == '__main__':
#     # re1 = "int|[a-zA-Z][a-zA-Z0-9]*|[0-9]+"
#     # re2 = "int|[a-z](0|01)*a|[0-9]+"

#     # re3 = "int|[a-z\\s-\\\\]*"
#     # re4 = r"int|[a-z\s-\\]*"
#     # re5 = '"\n[a\\s-\\t"]'
#     # re6 = r"a?([a-z\s-\\]*)?b+"
#     # re7 = "(int|1)"
#     # re8 = "[^0-9]?"

#     # re9 = "(int | 1)2)"
#     # re10 = "int |(1"

#     # re = RE(re9)
#     # if re.gt:
#     #     re.to_nfa()
#     #     re.dump_nfa_dot()
    
#     pattern = r'int|[a-zA-Z][a-zA-Z0-9]*|[0-9]+'
#     text_list = [r'int', r'inta', r'1a', r'20348']
#     re_obj = RE(pattern)
#     if re_obj.gt:
#         re_obj.to_nfa()
#         # re.dump_nfa_dot()
#         for text in text_list:
#             ret = re_obj.match_with_nfa(text)
#             print('match result: ' + str(ret))


# test nfa_to_dfa & match_with_dfa
if __name__ == '__main__':
    # re1 = "int|[a-zA-Z][a-zA-Z0-9]*|[0-9]+"
    # re2 = "int|[a-z](0|01)*a|[0-9]+"

    # re3 = "int|[a-z\\s-\\\\]*"
    # re4 = r"int|[a-z\s-\\]*"
    # re5 = '"\n[a\\s-\\t"]'  # 这里写错了，应该是: \t-\s
    # re6 = r"a?([a-z\s-\\]*)?b+"
    # re7 = "(int|1)"
    # re8 = "[^0-9]?"

    # re9 = "(int | 1)2)"
    # re10 = "int |(1"

    # re = RE(re2)
    # re.to_nfa()
    # # re.dump_nfa_dot()
    # re.to_dfa()
    # re.dump_dfa_dot()
    
    pattern = r'int|[a-zA-Z][a-zA-Z0-9]*|[0-9]+'
    text_list = [r'int', r'inta', r'1a', r'20348']
    re_obj = RE(pattern)
    re_obj.to_nfa()
    # re.dump_nfa_dot()
    re_obj.to_dfa()
    # re.dump_dfa_dot()
    for text in text_list:
        ret = re_obj.match_with_dfa(text)
        print('match result: ' + str(ret))
