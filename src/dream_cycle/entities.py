"""
Dream Cycle — Entity extraction — topic keywords, keyword extraction, LLM-based, fallback chain
"""

import re
import json
import logging
from collections import Counter
from dream_cycle.config import log
from dream_cycle.llm import _call_infini

def extract_topic_key(text: str) -> str:
    """
    从记忆文本提取主题键 — 用于实体级聚类
    
    策略:
    1. 提取项目名/技能名/仓库名等实体
    2. 提取核心动作词
    3. 组合为主题键
    """
    import re
    
    # 项目/仓库名模式
    repo_patterns = [
        r'(hermes-config|hermes-agent|skills-vendors|vault|mem0-stack|neo4j-playground|quinnpm|Server-Admin|bondTickAnalysis|cv|memory-bridge|Vyakarana)',
        r'([\w-]+)/(?:skills?|repo|project)',
    ]
    
    # 技能名模式
    skill_patterns = [
        r'(?:skill|技能)[s]?\s*[:/]?\s*([a-z][\w-]+)',
        r'([a-z][\w-]+)/(?:SKILL|skill)',
    ]
    
    # 话题关键词
    topic_patterns = [
        r'(?:RBA|Fed|ECB|BOJ|PBOC)\s*(?:framework|decision|rate)',
        r'(?:Bloomberg|terminal|AI)\s*(?:prompt|query|research)',
        r'(?:Vault|wiki|Obsidian)\s*(?:cleanup|restructure|ingest|lint)',
        r'(?:Neo4j|graph)\s*(?:label|query|sync|entity)',
        r'(?:Docker|container)\s*(?:restart|build|deploy|health)',
        r'(?:mem0|memory)\s*(?:plugin|upgrade|model|auth)',
        r'(?:dream|梦)\s*(?:cycle|循环)',
        r'(?:CGB|UST|yield|bond)\s*(?:curve|spread|data)',
        r'(?:COMEX|silver|gold)\s*(?:delivery|inventory)',
    ]
    
    # 排除列表 — 常见误匹配
    EXCLUDE_ENTITIES = {'on', 'from', 'the', 'with', 'for', 'and', 'not', 'but', 'or', 'in', 'at', 'to', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their'}
    
    entities = []
    
    # 提取仓库名
    for p in repo_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            entities.append(m.group(1).lower())
    
    # 提取技能名
    for p in skill_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            entities.append(m.group(1).lower())
    
    # 提取话题
    for p in topic_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            entities.append(m.group(0).lower().replace(' ', '_'))
    
    if entities:
        filtered = [e for e in set(entities) if e.lower() not in EXCLUDE_ENTITIES and len(e) > 2]
        if filtered:
            return '|'.join(sorted(filtered))
    return ''


# ─── 高质量关键词提取 (替代原始 Counter) ──────────────────────────────

# 综合停用词表 (英文+中文+LLM常见垃圾)
_KEYWORD_STOP_WORDS = frozenset({
    # English
    'about', 'above', 'after', 'again', 'all', 'also', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'could', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'get', 'got', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
    'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more',
    'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off',
    'on', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
    'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such',
    'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
    'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    # LLM slop markers
    'user', 'assistant', 'system', 'model', 'output', 'input', 'response',
    'request', 'message', 'content', 'text', 'information', 'note',
    'however', 'therefore', 'thus', 'moreover', 'furthermore', 'additionally',
    'specifically', 'indeed', 'essentially', 'particularly', 'notably',
    'significantly', 'simply', 'actually', 'basically', 'literally',
    'obviously', 'clearly', 'delve', 'regarding', 'ensure', 'leverage',
    # Generic non-entity terms
    'trade', 'framework', 'data', 'system', 'project', 'file', 'update',
    'change', 'feature', 'lines', 'order', 'blocking', 'repository',
    'added', 'updated', 'created', 'removed', 'fixed', 'skill', 'new',
    'old', 'first', 'last', 'only', 'another', 'different', 'important',
    'using', 'based', 'need', 'make', 'like', 'know', 'think', 'want',
    'well', 'much', 'many', 'still', 'even', 'back', 'way', 'thing',
    'things', 'something', 'everything', 'nothing', 'anything', 'already',
    'always', 'never', 'every', 'without', 'within', 'along', 'since',
    # Common verbs (not entities)
    'implemented', 'uses', 'used', 'using', 'created', 'updated', 'removed',
    'deleted', 'added', 'fixed', 'changed', 'modified', 'replaced', 'merged',
    'installed', 'configured', 'deployed', 'started', 'stopped', 'running',
    'working', 'worked', 'requires', 'required', 'supports', 'supported',
    'provides', 'provided', 'includes', 'included', 'contains', 'contained',
    'allows', 'allowed', 'enables', 'enabled', 'prevents', 'prevented',
    'follows', 'followed', 'returns', 'returned', 'accepts', 'accepted',
    'handles', 'handled', 'processes', 'processed', 'generates', 'generated',
    # Common generic nouns (not entities)
    'memory', 'limit', 'service', 'command', 'option', 'parameter', 'value',
    'result', 'output', 'error', 'warning', 'status', 'version', 'number',
    'default', 'method', 'function', 'variable', 'argument', 'example',
    'format', 'type', 'name', 'path', 'directory', 'folder', 'script',
    'code', 'line', 'step', 'process', 'task', 'action', 'check', 'test',
    'access', 'rule', 'entry', 'table', 'column', 'row', 'field', 'key',
    'source', 'target', 'source', 'group', 'item', 'element', 'section',
    'block', 'module', 'component', 'instance', 'resource', 'record',
    # More generic verbs/nouns
    'reduced', 'usage', 'commits', 'commit', 'branch', 'remote', 'local',
    'cache', 'index', 'query', 'response', 'request', 'session', 'client',
    'server', 'host', 'port', 'domain', 'network', 'connection', 'timeout',
    'buffer', 'stream', 'batch', 'chunk', 'payload', 'header', 'token',
    'callback', 'handler', 'listener', 'observer', 'filter', 'mapper',
    'reducer', 'transform', 'convert', 'parse', 'validate', 'encode',
    'decode', 'serialize', 'deserialize', 'normalize', 'sanitize',
    # -ing forms (almost never entities)
    'adding', 'removing', 'updating', 'creating', 'deleting', 'running',
    'loading', 'saving', 'reading', 'writing', 'processing', 'handling',
    'checking', 'testing', 'building', 'deploying', 'starting', 'stopping',
    'monitoring', 'tracking', 'logging', 'reporting', 'notifying',
    'reducing', 'increasing', 'decreasing', 'improving', 'extending',
    'merging', 'splitting', 'moving', 'renaming', 'copying', 'pasting',
    'skipping', 'falling', 'rising', 'dropping', 'changing', 'setting',
    'getting', 'putting', 'calling', 'returning', 'showing', 'hiding',
    # Years
    '2024', '2025', '2026', '2027',
    # Chinese
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
    '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
    '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它',
    '那', '被', '从', '把', '让', '用', '对', '为', '与', '而',
})

# 领域关键词加权 — 投资术语 3x，技术术语 1.5x
_KEYWORD_DOMAIN_BOOST = {
    # Investment (3x)
    'bonds': 3, 'yield': 3, 'spread': 3, 'CGB': 3, 'UST': 3, 'carry': 3,
    'duration': 3, 'credit': 3, 'curve': 3, 'swap': 3, 'basis': 3,
    'delivery': 3, 'inventory': 3, 'bond': 3, 'rate': 3, 'inflation': 3,
    'fed': 3, 'ecb': 3, 'boj': 3, 'rb': 3, 'pboc': 3, 'macro': 3,
    'fiscal': 3, 'monetary': 3, 'hedge': 3, 'position': 3, 'flow': 3,
    'premium': 3, 'sovereign': 3, 'cme': 3, 'comex': 3, 'silver': 3,
    'gold': 3, 'treasury': 3, 'coupon': 3, 'maturity': 3, 'issuance': 3,
    '利差': 3, '利率': 3, '收益率': 3, '曲线': 3, '债券': 3, '信用': 3,
    '央行': 3, '通胀': 3, '利差': 3, '溢价': 3, '对冲': 3, '仓位': 3,
    # Tech (1.5x)
    'docker': 1.5, 'plugin': 1.5, 'mcp': 1.5, 'mem0': 1.5, 'neo4j': 1.5,
    'config': 1.5, 'deploy': 1.5, 'cron': 1.5, 'api': 1.5, 'hermes': 1.5,
    'vault': 1.5, 'ssh': 1.5, 'tunnel': 1.5, 'token': 1.5, 'oauth': 1.5,
}

# 有意义的实体最小长度
_ENTITY_MIN_LEN = 4


def extract_keywords(texts: list[str], top_n: int = 5, min_score: float = 1.0) -> list[str]:
    """
    从文本列表中提取高质量关键词 — 替代原始 Counter 词频统计
    
    改进:
    1. 综合停用词过滤 (英文+中文+LLM slop)
    2. 标点/URL/代码块清洗
    3. Bigram 短语检测 (两词组合)
    4. 领域感知加权 (投资/技术术语优先)
    5. 缩写识别 (全大写2-5字母)
    6. 最低频率/分数阈值
    """
    import re
    from collections import Counter
    
    combined = " ".join(texts)
    
    # 标点清洗: 去代码块、URL、markdown
    cleaned = re.sub(r'```[\s\S]*?```', '', combined)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = re.sub(r'[^\w\s\u4e00-\u9fff-]', ' ', cleaned)
    
    words = cleaned.split()
    
    # 单词级提取 + 加权
    word_scores: Counter = Counter()
    for w in words:
        w_clean = w.lower().rstrip(',.').strip('-')
        if not w_clean or len(w_clean) < _ENTITY_MIN_LEN:
            continue
        if w_clean in _KEYWORD_STOP_WORDS:
            continue
        if w_clean.isdigit():
            continue
        # 通用 -ing 检测: 任何以 -ing 结尾且不在 domain boost 中的词，99%不是实体
        if w_clean.endswith('ing') and w_clean not in _KEYWORD_DOMAIN_BOOST:
            continue
        
        # 域加权
        boost = _KEYWORD_DOMAIN_BOOST.get(w_clean, 1.0)
        # 缩写加分 (全大写2-5字母 = 可能是重要缩写)
        if w.isupper() and 2 <= len(w) <= 5:
            boost = max(boost, 2.0)
        
        word_scores[w_clean] += boost
    
    # Bigram 检测
    for i in range(len(words) - 1):
        w1 = words[i].lower().rstrip(',.').strip('-')
        w2 = words[i + 1].lower().rstrip(',.').strip('-')
        if (len(w1) >= _ENTITY_MIN_LEN and len(w2) >= _ENTITY_MIN_LEN
                and w1 not in _KEYWORD_STOP_WORDS and w2 not in _KEYWORD_STOP_WORDS
                and not w1.isdigit() and not w2.isdigit()):
            bigram = f"{w1}-{w2}"
            word_scores[bigram] += 1.5
    
    # 过滤 + 排序
    filtered = {w: s for w, s in word_scores.items() if s >= min_score}
    sorted_kw = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    
    return [w for w, _ in sorted_kw[:top_n]]


def llm_extract_entities(texts: list[str], max_entities: int = 5) -> list[str] | None:
    """
    用 LLM 从文本中提取高质量实体 — 替代规则式关键词提取
    
    LLM 理解语义，不会把 'assistant', '2026', 'reducing' 当实体。
    一次调用处理一个 cluster 的所有文本，成本约 300 tokens。
    
    Returns: 实体名列表 or None (API 失败时 fallback 到 extract_keywords)
    """
    combined = "\n".join(f"- {t[:300]}" for i, t in enumerate(texts[:10]))
    prompt = f"""从以下记忆中提取有价值的实体名。只提取真正的专有名词：项目名、工具名、技术名、组织名、金融术语、数据源名。

不要提取：通用动词/名词(user/assistant/system/running/adding)、年份(2026)、停用词、-ing 形式。

记忆内容:
{combined}

只输出 JSON 数组，最多{max_entities}个实体，按重要性排序:
["实体1", "实体2", ...]

无有价值实体时输出: []"""

    result = _call_infini(prompt, max_tokens=200, temperature=0.1)
    if not result:
        return None
    
    # 提取 JSON 数组
    import re
    json_match = re.search(r'\[[\s\S]*?\]', result)
    if not json_match:
        return None
    
    try:
        entities = json.loads(json_match.group())
        if isinstance(entities, list):
            # 基础清洗：空字符串/纯数字/太短的
            clean = [e.strip() for e in entities 
                     if isinstance(e, str) and len(e.strip()) >= 2 
                     and not e.strip().isdigit()]
            return clean[:max_entities] if clean else None
    except json.JSONDecodeError:
        pass
    return None


def extract_entities_with_fallback(texts: list[str], max_entities: int = 5) -> list[str]:
    """
    实体提取：LLM 优先，规则 fallback
    
    LLM 提取的实体质量远高于规则（理解语义、不会提取垃圾词），
    但 API 可能失败（429/超时），此时降级到 extract_keywords()。
    """
    # LLM 优先
    entities = llm_extract_entities(texts, max_entities=max_entities)
    if entities:
        log.info(f"  🤖 LLM 实体提取: {entities}")
        return entities
    
    # Fallback: 规则提取
    log.info(f"  ⚙️ LLM 失败, fallback 到规则提取")
    return extract_keywords(texts, top_n=max_entities, min_score=1.0)


def _is_valid_entity(name: str) -> bool:
    """判断一个字符串是否是有效的实体名 (非停用词、非纯数字、有实际含义)"""
    if not name or len(name) < 2:
        return False
    name_lower = name.lower().rstrip(',.').strip('-')
    if not name_lower:
        return False
    if name_lower in _KEYWORD_STOP_WORDS:
        return False
    if name_lower.isdigit():
        return False
    if name_lower.replace('-', '').isdigit():
        return False
    # 全是同一个字符
    if len(set(name_lower.replace('-', ''))) == 1:
        return False
    # 太短(<=3)且不是已知缩写 → 跳过
    if len(name_lower) <= 3 and name_lower not in _KEYWORD_DOMAIN_BOOST and not name.isupper():
        return False
    # -ing 结尾且不在 domain boost → 几乎不可能是实体
    if name_lower.endswith('ing') and name_lower not in _KEYWORD_DOMAIN_BOOST:
        return False
    return True


