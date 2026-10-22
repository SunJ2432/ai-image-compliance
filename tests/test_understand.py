import pytest
from imgcompliance.captioner import Captioner, parse_caption_json
from imgcompliance.classifier import Classifier

VALID = '{"scene_summary":"A wooded path","depicts_people":false,"appears_photorealistic":true,"possible_real_world_reference":null}'


def test_json_parser_accepts_fence():
    result = parse_caption_json('```json\n' + VALID + '\n```')
    assert result['depicts_people'] is False
    assert result['possible_real_world_reference'] is None


def test_retry_once():
    calls = []
    def fake(image, prompt):
        calls.append(prompt)
        return 'invalid' if len(calls) == 1 else VALID
    result = Captioner(generator=fake).describe('mock-image')
    assert len(calls) == 2
    assert result['format'] == 'json'
    assert 'NEVER identify' in calls[0]


def test_plain_text_fallback():
    calls = []
    def fake(image, prompt):
        calls.append(prompt)
        return 'A tree-lined stone bridge.'
    result = Captioner(generator=fake).describe('mock-image')
    assert len(calls) == 2
    assert result['format'] == 'plain_text_fallback'
    assert result['scene_summary'] == 'A tree-lined stone bridge.'
    assert result['depicts_people'] is None


@pytest.mark.parametrize('purpose', [None, 'news', 'education', 'Art'])
def test_creative_rejected_without_valid_purpose(purpose):
    fake = lambda payload: '{"subclass":"deepfake_creative","rationale":"Creative image."}'
    result = Classifier(generator=fake).classify({}, {'label': 'ai'}, {'verified_real_entity': True, 'declared_purpose': purpose})
    assert result['subclass'] == 'general_synthetic'


@pytest.mark.parametrize('purpose', ['art', 'satire'])
def test_creative_allowed_with_declared_purpose_and_evidence(purpose):
    fake = lambda payload: '{"subclass":"deepfake_creative","rationale":"Declared creative use."}'
    result = Classifier(generator=fake).classify({}, {'label': 'ai'}, {'verified_real_entity': True, 'declared_purpose': purpose})
    assert result['subclass'] == 'deepfake_creative'


def test_creative_rejected_without_verified_entity():
    fake = lambda payload: '{"subclass":"deepfake_creative","rationale":"Art."}'
    result = Classifier(generator=fake).classify({}, {'label': 'ai'}, {'declared_purpose': 'art'})
    assert result['subclass'] == 'general_synthetic'


def test_real_short_circuits_model():
    result = Classifier(generator=lambda _: pytest.fail('should not call model')).classify({}, {'label': 'real'})
    assert result['subclass'] == 'not_ai_generated'


def test_invalid_classifier_response_fallback():
    result = Classifier(generator=lambda _: 'not JSON').classify({}, {'label': 'ai'})
    assert result['subclass'] == 'general_synthetic'


def test_no_real_entity_without_verification():
    fake = lambda _: '{"subclass":"deepfake_real_entity","rationale":"Claimed real."}'
    result = Classifier(generator=fake).classify({}, {'label': 'ai'}, {})
    assert result['subclass'] == 'general_synthetic'
