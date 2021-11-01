# 一个简单的正则表达式引擎
功能：仅支持如下元字符：( ) [ ] ^ - | + * ?
<br/></br>

## 介绍
* 手写正则表达式的`parser`，能够自动解析`re`字符串，并生成`AST`
* 再通过《编译原理之美》课程的知识，基于`AST`生成`NFA`，基于`NFA`生成`DFA`，以此实现类似`re.match`的功能
<br/></br>

## 示例程序
```python
from my_re import RE

pattern = r'int|[a-zA-Z][a-zA-Z0-9]*|[0-9]+'
text_list = [r'int', r'inta', r'1a', r'20348']
re = RE(pattern)
# re.gt.dump_dot()
re.to_nfa()
# re.dump_nfa_dot()
re.to_dfa()
# re.dump_dfa_dot()
for text in text_list:
    ret = re_obj.match_with_dfa(text)
    print('match result: ' + str(ret))
```
* 代码设计上，侧重于学习目的，而非实用目的，所以使用方式上不太友好
* 某个`re`对应的`AST`、`NFA`、`DFA`，可以分别使用：`re.gt.dump_dot()`、`re.dump_nfa_dot()`、`re.dump_dfa_dot()`，将其图形按`DOT`格式打印出来，并用`Graphviz`工具查看
<br/></br>

## 与实用RE引擎的区别
* 此`RE`引擎一切从简，仅实现了`RE`功能的一个最小集
* 仅支持的转义字符：`\s \t \r \n \\ \+ \* \? \- \( \) \[ \]`
* 不支持点号通配符`.`
* 不支持特殊字符集，如：`\w \W \d \D \s` 等
* 不支持花括号量词，如：`{1, 3}`
* 不支持捕获和引用
* 默认按照贪婪策略进行匹配，不支持“忽略优先”的匹配策略，即：`+?  *?  ??`
* 不支持断言和环视，如：`\b  ^  $  (?<=x)  (?<!x)`
* 不支持`Unicode`
* 不支持设定特殊的匹配模式，如：`Case-Insensitive、Multiline`