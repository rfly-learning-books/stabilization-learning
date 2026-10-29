"""Reviewed checklist-to-implementation map; no substitute application claims."""
from pathlib import Path
from .queue import PROJECT, read, sha, write

CHECKLIST = Path("E:/rfly-books/output/reports/全书实验验证任务清单_20261028/实验任务清单.json")


MAPPING = {
    "E00": ("queue_protocol_and_status_matrix", "32个原任务逐项留账；每个新实验单独冻结来源、预算和数据"),
    "E01": ("scripts/book_checks/run_book_checks.py", "已有教学数学复算；频率兼容入口独立，不能作为应用训练验收"),
    "E02": (None, "未找到同协议跟踪/内模方法完整对照入口"),
    "E03": ("scripts/run_book_disk_overnight_v1.py", "Reviewed固定接触单侧圆盘，known/direct-D/identified/GMM四路线"),
    "E04": ("scripts/run_book_disk_overnight_v1.py", "圆盘平均拟合、误差裕度和反例增强，完整后继审计；应用另记"),
    "E05": ("scripts/run_book_disk_overnight_v1.py", "共享圆盘的数据量/模型误差子证据；完整激励和辨识扫描尚需单独验收"),
    "E06": (None, "L学习/机械结构先验不能由现有M学习或MuJoCo推送替代"),
    "E07": ("configs/book_multiseed_v4.json", "历史四物理基底M训练可复用，不含核/积分候选完整消融；原入口强制5种子"),
    "E08": (None, "无输出历史/积分增广统一窗口实验入口"),
    "E09": ("scripts/book_checks/run_disk_multiseed_book_v4.py", "已有8000/4000独立D回归与MC；新增e09命令补乘性/持续加性噪声和采样周期"),
    "E10": ("scripts/run_book_grid_overnight_v1.py", "Reviewed95状态MDP，三策略独立回合、模型传播、约束/滑移/时域扫描"),
    "E11": (None, "几何Push Full-M不替代视觉接触特征四类消融"),
    "E12": (None, "分别训练RGB/Push/观测不构成共享表示迁移矩阵"),
    "E13": ("scripts/run_book_rl_overnight_v1.py", "同预算SAC纯奖励/软下降/硬过滤解析2D积分器子实验；完整应用公平RL对照另记"),
    "E14": ("scripts/run_unified_training_v4.py", "此次观测与E20继续按env/algo/runner验收；全支持矩阵和隔离安装另记"),
    "E15": (None, "普通epoch选模不是在线连续证书切换/回退实验"),
    "E16": ("scripts/run_rgb_book_reproduction_v1.py", "当前AE-RGB书稿训练结果单列；不能等同第13章B/J/P/G前视四通道对照"),
    "E17": (None, "当前固定FastDepth不能替代深度换源独立实验"),
    "E18": (None, "下视实际光流与解析稳健拟合未接入本队列"),
    "E19": (None, "无人机完整执行链/实飞需硬件，不以RGB仿真替代"),
    "E20": ("scripts/run_book_overnight_v1.py e20", "书稿原A/B+同初始新seed41 C，8形状响应与MuJoCo子步保持分层统计"),
    "E21": (None, "实物机械臂验证需真实设备；E20只有MuJoCo"),
    "E22": ("scripts/run_book_overnight_v1.py observers", "v4合作目标13存储坐标/12维切空间、几何退化与噪声；合成关键点前端，不含检测器"),
    "E23": (None, "已知目标位姿不能替代未知场景自身运动/尺度恢复"),
    "E24": (None, "车摆已知输入观测不是估计器与控制器联合反馈闭环"),
    "E25": (None, "主代理另行搭建真实冻结生成模型；二次解析代理不算图像生成对照"),
    "E26": (None, "无未知动力学安全差分/视场完整应用入口"),
    "E27": (None, "多机协同和虚拟管道不能由单机仿真外推"),
    "E28": (None, "流匹配规划与真实执行链尚未实施"),
    "E29": (None, "数值求根/镇定优化扫描尚无本队列入口"),
    "E30": (None, "吸引子恢复率/错误吸引域扫描未实施"),
    "E31": (None, "有限MDP不替代离散博弈/固定对手与对抗评估"),
}


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    checklist = read(CHECKLIST)
    rows = []
    for original in checklist["tasks"]:
        entry, note = MAPPING[original["id"]]
        physical_entry = PROJECT / entry.split(" ")[0] if entry and "/" in entry else None
        rows.append(dict(original_task=original, mapped_entry=entry,
            entry_exists=physical_entry.is_file() if physical_entry else None,
            implementation_scope=note, completed=False, full_task_acceptance=False))
    output.mkdir(parents=True)
    result = dict(schema="reviewed_book_asset_mapping_v1", checklist=str(CHECKLIST), checklist_sha256=sha(CHECKLIST),
        reviewed_source_root=checklist["source_root"], task_count=len(rows), entries=rows)
    write(output / "ASSET_MAPPING.json", result)
    lines = ["# Reviewed全书任务与本机执行入口", "", "本表记录已核实入口和证据范围，完成情况以新运行结果为准。", "",
             "|任务|入口|范围|", "|---|---|---|"]
    lines += [f"|{r['original_task']['id']} {r['original_task']['title']}|{r['mapped_entry'] or '待实施'}|{r['implementation_scope']}|" for r in rows]
    (output / "ASSET_MAPPING.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return dict(status="prepared", output=str(output), tasks=len(rows))
