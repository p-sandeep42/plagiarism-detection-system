import ast
import re
try:
    import esprima
except ImportError:
    esprima = None

class StructuralVisitor(ast.NodeVisitor):
    def __init__(self):
        self.structure = []
    def generic_visit(self, node):
        self.structure.append(type(node).__name__)
        super().generic_visit(node)

def get_python_structure(code: str) -> str:
    try:
        # Prevent evaluation of malicious Python strings
        tree = ast.parse(code)
        visitor = StructuralVisitor()
        visitor.visit(tree)
        return " ".join(visitor.structure)
    except Exception:
        return "ERROR_PARSING_PYTHON"

def get_js_structure(code: str) -> str:
    if esprima is None:
        return ""
    try:
        tree = esprima.parseScript(code, {"loc": False, "range": False})
        
        # Simple recursive function to extract node types safely
        def traverse(node):
            types = []
            if isinstance(node, dict):
                if 'type' in node: types.append(node['type'])
                for key, val in node.items():
                    types.extend(traverse(val))
            elif isinstance(node, list):
                for item in node:
                    types.extend(traverse(item))
            elif hasattr(node, "type"): 
                types.append(node.type)
                # rudimentary fallback for esprima nodes
                try:
                    for k in dir(node):
                        if not k.startswith("_") and k != "type":
                            types.extend(traverse(getattr(node, k)))
                except:
                    pass
            return types
            
        structure_types = traverse(tree.to_dict() if hasattr(tree, "to_dict") else tree)
        return " ".join(structure_types)
    except Exception:
        return "ERROR_PARSING_JS"

def jaccard_similarity(text1: str, text2: str) -> float:
    def tokenize(t): 
        return set(re.findall(r'\w+', t.lower()))
    
    set1 = tokenize(text1)
    set2 = tokenize(text2)
    
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
        
    return len(set1.intersection(set2)) / len(set1.union(set2))

def structural_similarity(text1: str, text2: str, ext: str) -> float:
    """
    If file is code, compare AST structure instead of raw text tokens.
    """
    ext = ext.lower().lstrip(".")
    if ext == "py":
        s1 = get_python_structure(text1)
        s2 = get_python_structure(text2)
        return jaccard_similarity(s1, s2)
    elif ext == "js":
        s1 = get_js_structure(text1)
        s2 = get_js_structure(text2)
        return jaccard_similarity(s1, s2)
    else:
        # For non-code documents, structural similarity can just be Jaccard over paragraphs/sentences or words.
        return jaccard_similarity(text1, text2)
