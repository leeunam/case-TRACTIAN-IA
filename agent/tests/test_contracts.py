import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from tractian_agent.contracts import (
    ApiError,
    ApiErrorCategory,
    ApiResult,
    ActionReceipt,
    Identity,
    ResponseMode,
    SupportRequest,
    ToolCall,
)


class GetAssetArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    include: list[str]


def test_identity_preserves_trusted_context():
    identity = Identity(
        user_id="usr_pedro",
        company_id="comp_mineracao_andes",
    )

    assert identity.user_id == "usr_pedro"
    assert identity.company_id == "comp_mineracao_andes"


def test_identity_rejects_empty_user_id():
    with pytest.raises(ValidationError):
        Identity(
            user_id="",
            company_id="comp_mineracao_andes",
        )


def test_identity_rejects_empty_company_id():
    with pytest.raises(ValidationError):
        Identity(
            user_id="usr_pedro",
            company_id="",
        )


def test_identity_rejects_blank_user_id():
    with pytest.raises(ValidationError):
        Identity(
            user_id="   ",
            company_id="comp_mineracao_andes",
        )


def test_identity_rejects_blank_company_id():
    with pytest.raises(ValidationError):
        Identity(
            user_id="usr_pedro",
            company_id="   ",
        )


def test_identity_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
            expected_path=["dado proibido"],
        )


def test_support_request_preserves_message_and_trusted_identity():
    request = SupportRequest(
        case_id="case_tkt_inv_04",
        ticket_id="TKT-INV-04",
        asset_id="asset_G501",
        message=(
            "O redutor da correia transportadora quebrou ontem e eu não recebi "
            "nenhum aviso. Por quê?"
        ),
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )

    assert request.case_id == "case_tkt_inv_04"
    assert request.identity.user_id == "usr_pedro"


def test_support_request_accepts_case_without_central_asset():
    request = SupportRequest(
        case_id="case_kb_01",
        ticket_id="TKT-KB-01",
        asset_id=None,
        message="Como funciona o cálculo de baseline?",
        identity=Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    )

    assert request.asset_id is None


def test_support_request_requires_asset_id_field_even_when_nullable():
    with pytest.raises(ValidationError):
        SupportRequest(
            case_id="case_kb_01",
            ticket_id="TKT-KB-01",
            message="Como funciona o cálculo de baseline?",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
        )


def test_support_request_rejects_blank_message():
    with pytest.raises(ValidationError):
        SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="   ",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
        )


@pytest.mark.parametrize("field", ["case_id", "ticket_id", "asset_id"])
def test_support_request_rejects_blank_identifiers(field):
    request_data = {
        "case_id": "case_tkt_inv_04",
        "ticket_id": "TKT-INV-04",
        "asset_id": "asset_G501",
        "message": "Por que não recebi nenhum aviso?",
        "identity": Identity(
            user_id="usr_pedro",
            company_id="comp_mineracao_andes",
        ),
    }
    request_data[field] = "   "

    with pytest.raises(ValidationError):
        SupportRequest(**request_data)


def test_support_request_rejects_golden_set_fields():
    with pytest.raises(ValidationError):
        SupportRequest(
            case_id="case_tkt_inv_04",
            ticket_id="TKT-INV-04",
            asset_id="asset_G501",
            message="Por que não recebi nenhum aviso?",
            identity=Identity(
                user_id="usr_pedro",
                company_id="comp_mineracao_andes",
            ),
            expected_path=[{"step": "GET /assets/asset_G501"}],
        )


def test_tool_call_preserves_json_arguments():
    call = ToolCall[GetAssetArguments](
        call_id="call_01",
        name="get_asset",
        arguments=GetAssetArguments(
            asset_id="asset_G501",
            include=["points"],
        ),
    )

    assert call.name == "get_asset"
    assert call.arguments.asset_id == "asset_G501"


def test_tool_call_rejects_non_json_arguments():
    with pytest.raises(ValidationError):
        ToolCall[GetAssetArguments](
            call_id="call_01",
            name="get_asset",
            arguments={"invalid": object()},
        )


def test_tool_call_rejects_identity_outside_tool_argument_schema():
    with pytest.raises(ValidationError):
        ToolCall[GetAssetArguments](
            call_id="call_01",
            name="get_asset",
            arguments={
                "asset_id": "asset_G501",
                "include": ["points"],
                "user_id": "usr_fabricado_pelo_llm",
            },
        )


def test_api_result_preserves_typed_data_and_query_mode():
    result = ApiResult[dict[str, str]](
        status_code=200,
        data={"id": "asset_G501"},
        mode=ResponseMode.COMPLETE,
    )

    assert result.ok is True
    assert result.data["id"] == "asset_G501"
    assert result.mode is ResponseMode.COMPLETE


def test_api_error_preserves_category_code_and_status():
    error = ApiError(
        category=ApiErrorCategory.API,
        code="NOT_FOUND",
        message="Ativo não encontrado.",
        status_code=404,
    )

    assert error.ok is False
    assert error.category is ApiErrorCategory.API
    assert error.status_code == 404


def test_action_receipt_preserves_api_confirmation():
    receipt = ActionReceipt(
        accepted=True,
        action_id="act_1234abcd",
        message="Reprocesso aceito.",
    )

    assert receipt.accepted is True
    assert receipt.action_id == "act_1234abcd"
