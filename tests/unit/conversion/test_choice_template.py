"""Contracts for the multiple-choice card templates."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

TEMPLATE_DIR = Path(__file__).parents[3] / "ankiops/default_note_types/AnkiOpsChoice"


def _render_template(filename: str, fields: dict[str, str]) -> str:
    template = (TEMPLATE_DIR / filename).read_text(encoding="utf-8")
    # The choice sections in the front and back templates are not nested.
    template = re.sub(
        r"{{#(Choice \d)}}(.*?){{/\1}}",
        lambda match: match.group(2) if fields.get(match.group(1)) else "",
        template,
        flags=re.DOTALL,
    )
    for name, value in fields.items():
        template = template.replace("{{" + name + "}}", value)
    return template


@pytest.mark.parametrize("filename", ["Front.template.anki", "Back.template.anki"])
def test_choices_are_html_fields_not_javascript_literals(filename: str) -> None:
    template = (TEMPLATE_DIR / filename).read_text(encoding="utf-8")
    html, script = template.split("<script>", 1)
    for num in range(1, 9):
        field_tag = f"{{{{Choice {num}}}}}"
        field_element = f'<div class="choice-item" data-num="{num}">{field_tag}</div>'
        assert field_element in html
        assert field_tag not in script
    assert "container.appendChild(choice.element)" in script
    assert "div.innerHTML = choice.text" not in script


def test_shuffling_preserves_content_and_matches_back() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed to execute the card template scripts")

    fields = {
        "Question": "A question",
        "Answer": "1, 3",
        "Choice 1": r"Math: \(\frac{1}{2}\)",
        "Choice 2": 'Quotes " and ` ${example} and <b>bold</b>',
        "Choice 3": r"Display: \[x^2\]",
    }
    sides = []
    for filename in ("Front.template.anki", "Back.template.anki"):
        rendered = _render_template(filename, fields)
        soup = BeautifulSoup(rendered, "html.parser")
        choices = soup.select("#choices > .choice-item")
        sides.append(
            {
                "script": soup.script.string,
                "choices": [
                    {"num": element["data-num"], "html": element.decode_contents()}
                    for element in choices
                ],
            }
        )

    runner = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const storage = {};
const localStorage = {
  getItem: key => storage[key] || null,
  setItem: (key, value) => { storage[key] = value; }
};
function run(side) {
  const hint = {textContent: ''};
  const container = {
    children: side.choices.map(item => ({
      innerHTML: item.html,
      getAttribute: () => item.num,
      classList: {values: [], add(value) { this.values.push(value); }}
    })),
    appendChild(element) {
      this.children.splice(this.children.indexOf(element), 1);
      this.children.push(element);
    }
  };
  const document = {
    documentElement: {dataset: {cid: '123'}},
    getElementById: id => id === 'choices' ? container : hint
  };
  new Function('document', 'localStorage', side.script)(document, localStorage);
  return {hint: hint.textContent, choices: container.children.map(element => ({
    num: element.getAttribute('data-num'), html: element.innerHTML,
    classes: element.classList.values
  }))};
}
console.log(JSON.stringify(input.map(run)));
"""
    result = subprocess.run(
        [node, "-e", runner],
        input=json.dumps(sides),
        text=True,
        capture_output=True,
        check=True,
    )
    front, back = json.loads(result.stdout)
    assert front["hint"] == back["hint"] == "2 answers"
    assert [choice["num"] for choice in front["choices"]] == [
        choice["num"] for choice in back["choices"]
    ]
    for side in (front, back):
        assert {choice["num"]: choice["html"] for choice in side["choices"]} == {
            str(num): fields[f"Choice {num}"] for num in range(1, 4)
        }
    for choice in back["choices"]:
        expected = (
            "choice-correct" if choice["num"] in ("1", "3") else "choice-incorrect"
        )
        assert choice["classes"] == [expected]
