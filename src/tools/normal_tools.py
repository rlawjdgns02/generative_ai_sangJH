"""
normal_tools.py

일반 유틸 도구 모음:
- 계산기 (calculate)
- 날짜 차이 계산 (calculate_date_difference) -> 내부적으로 calculate 사용
"""

import ast
import operator
from typing import Dict, Any
from datetime import datetime

# 안전한 수식 계산을 위한 연산자 매핑
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    """AST 기반 안전 수식 계산"""
    if isinstance(node, ast.Constant):
        return node.value
    elif hasattr(ast, "Num") and isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"지원하지 않는 연산자: {type(node.op)}")
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op = SAFE_OPERATORS.get(type(node.op))
        if op is None:
            raise ValueError(f"지원하지 않는 연산자: {type(node.op)}")
        return op(operand)
    else:
        raise ValueError(f"지원하지 않는 표현식: {type(node)}")


def calculate(expression: str) -> Dict[str, Any]:
    """
    계산기 Tool 함수
    지원 연산: +, -, *, /, //, %, **, 괄호
    """
    try:
        expression = expression.strip()

        # 기본적인 숫자와 연산자만 허용 (보안)
        allowed_chars = set("0123456789+-*/.()% ")
        if not all(c in allowed_chars for c in expression):
            return {
                "expression": expression,
                "result": None,
                "ok": False,
                "error": "수식에는 숫자와 기본 연산자(+, -, *, /, %, **)만 사용할 수 있습니다.",
            }

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree.body)
            return {"expression": expression, "result": result, "ok": True}
        except SyntaxError as e:
            return {"expression": expression, "result": None, "ok": False, "error": f"수식 구문 오류: {str(e)}"}
        except ValueError as e:
            return {"expression": expression, "result": None, "ok": False, "error": f"계산 오류: {str(e)}"}
    except Exception as e:
        return {"expression": expression, "result": None, "ok": False, "error": f"계산 중 오류 발생: {str(e)}"}


def calculate_date_difference(date1: str, date2: str = None, unit: str = "days") -> Dict[str, Any]:
    """
    날짜 차이 계산 Tool 함수
    내부적으로 calculate 툴을 사용하여 연산을 수행합니다.
    """
    try:
        formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"]

        dt1 = None
        for fmt in formats:
            try:
                dt1 = datetime.strptime(date1, fmt)
                break
            except ValueError:
                continue

        if dt1 is None:
            return {
                "date1": date1,
                "date2": date2,
                "ok": False,
                "error": "date1 날짜 형식을 인식할 수 없습니다.",
            }

        if date2 is None:
            dt2 = datetime.now()
        else:
            dt2 = None
            for fmt in formats:
                try:
                    dt2 = datetime.strptime(date2, fmt)
                    break
                except ValueError:
                    continue
            if dt2 is None:
                return {
                    "date1": date1,
                    "date2": date2,
                    "ok": False,
                    "error": "date2 날짜 형식을 인식할 수 없습니다.",
                }

        ts1 = dt1.timestamp()
        ts2 = dt2.timestamp()

        unit_divisors = {
            "days": 86400,    
            "hours": 3600,    
            "minutes": 60,
            "seconds": 1
        }

        if unit not in unit_divisors:
             return {
                "ok": False,
                "error": "지원하지 않는 단위: days, hours, minutes, seconds",
            }
        
        divisor = unit_divisors[unit]

        expression = f"({ts2} - {ts1}) / {divisor}"

        calc_result = calculate(expression)

        if not calc_result["ok"]:
            return {
                "ok": False,
                "error": f"날짜 계산(calculate 호출) 중 오류: {calc_result.get('error')}"
            }

        difference = abs(calc_result["result"])

        return {
            "date1": date1, 
            "date2": date2 or "오늘", 
            "difference": difference, 
            "unit": unit, 
            "ok": True,
        }

    except Exception as e:
        return {
            "date1": date1, 
            "date2": date2, 
            "ok": False, 
            "error": f"날짜 계산 중 시스템 오류 발생: {str(e)}"
        }


# Tool 레지스트리
NORMAL_TOOLS = {
    "calculate": calculate,
    "calculate_date_difference": calculate_date_difference,
}