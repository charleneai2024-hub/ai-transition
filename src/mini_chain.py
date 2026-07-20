"""Three-stage chain demo: LLM extracts skills -> pure Python diffs against a
resume -> LLM explains the gaps and gives advice.

Stage 1 (LLM): pull 3-5 skill keywords out of a fake JD.
Stage 2 (pure Python): parse stage 1's JSON, diff against a hardcoded resume
skill list, find what the JD wants that the resume is missing.
Stage 3 (LLM): explain why each missing skill matters for an AI role, then
give encouraging suggestions in Chinese.
"""

import json
import sys

from llm_client import call_llm_messages

JD_TEXT = """岗位:AI应用工程师(数据分析师转型方向)
职责:基于业务数据构建机器学习/LLM应用原型,推动分析结果落地为AI产品功能。
要求:
1. 熟练使用Python进行数据处理与建模,熟悉pandas、scikit-learn等常用库;
2. 具备SQL数据查询与清洗能力,能独立完成特征工程;
3. 了解大语言模型(LLM)的Prompt设计与API调用,有相关项目经验优先;
4. 熟悉Git版本管理,具备良好的代码规范意识。"""

RESUME_SKILLS = ["Python", "Excel", "pandas"]


def stage1_extract_skills(jd_text: str) -> str:
    system = (
        "You are a precise information extraction engine. Extract 3-5 skill "
        "keywords from the job description the user provides. Output only a "
        'JSON string array, for example ["Python","SQL"]. Do not include any '
        "explanation, markdown code fences, or any other text."
    )
    return call_llm_messages(
        messages=[{"role": "user", "content": jd_text}],
        system=system,
    )


def stage2_find_missing_skills(raw_json: str, resume_skills: list[str]) -> list[str]:
    try:
        jd_skills = json.loads(raw_json)
    except json.JSONDecodeError:
        print("[stage2] failed to parse model output as JSON. Raw output was:")
        print(raw_json)
        sys.exit(1)

    jd_lower_to_original = {skill.lower(): skill for skill in jd_skills}
    resume_lower = {skill.lower() for skill in resume_skills}

    missing_lower = set(jd_lower_to_original) - resume_lower
    return [jd_lower_to_original[skill] for skill in missing_lower]


def stage3_explain_and_advise(missing_skills: list[str]) -> str:
    system = (
        "你是一位AI职业发展顾问。用户会给你一份缺失技能清单(JSON数组)。"
        "请先针对每个技能,用一句话解释它对AI岗位为什么重要;"
        "然后另起一段,用礼貌、鼓励的语气给出2-3句总体改进建议。"
        "全部使用中文输出,保持简洁,不要使用markdown标题、分隔线或大量项目符号。"
    )
    user_content = "缺失技能清单:" + json.dumps(missing_skills, ensure_ascii=False)
    return call_llm_messages(
        messages=[{"role": "user", "content": user_content}],
        system=system,
        max_tokens=600,
    )


def main() -> None:
    print("=" * 60)
    print("环1: 从JD提取技能关键词")
    print("=" * 60)
    stage1_output = stage1_extract_skills(JD_TEXT)
    print(stage1_output)

    print()
    print("=" * 60)
    print("环2: 对比JD技能与简历技能,找出缺失项")
    print("=" * 60)
    missing_skills = stage2_find_missing_skills(stage1_output, RESUME_SKILLS)
    print(f"简历技能: {RESUME_SKILLS}")
    print(f"缺失技能: {missing_skills}")

    print()
    print("=" * 60)
    print("环3: 解释缺失技能的重要性并给出改进建议")
    print("=" * 60)
    stage3_output = stage3_explain_and_advise(missing_skills)
    print(stage3_output)


if __name__ == "__main__":
    main()
