from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_core.documents.base import Document
from vectorstore import load_vector_from_local


@tool
def handle_transfer(input: str):
    """사용자의 등급(VIP/일반)과 이체 요청 금액에 따라 이체 가능 여부와 필요한 추가 인증에 관한 참고문서를 반환합니다.

    Args:
        input: 사용자 질의
    
    Returns:
        이체 가능 여부와 필요한 추가 인증에 관한 텍스트
    """
    vectorstore = load_vector_from_local()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {
            "k": 3
        }
    )
    chain = retriever | RunnableLambda(format_docs)

    context = chain.invoke(input)
    
    return context


@tool
def handle_block_transaction():
    """특정 거래가 차단되었을 때, 해당 차단의 근거가 되는 규제와 이로 인해 에이전트가 수행하게 될 다음 노드를 설명하는 참고문서를 반환한다.

    Returns:
        차단 근거와 다음 수행해야 할 노드 정보가 담긴 텍스트
    """
    
    vectorstore = load_vector_from_local()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {
            "k": 10
        }
    )
    filtered_docs = [doc for doc in retriever.invoke("id: 9, id: 10") 
                     if doc.metadata.get('id') in [9, 10]]
    
    context = format_docs(filtered_docs)

    return context

@tool
def transfer_loan_verification():
    """사용자가 해외 송금이나 대출을 요청할 때, **ID 26(해외한도), ID 30(DSR), ID 39(투자적합성)**를 순차적으로 검증하기 위한 참고 문서를 반환

    Returns:
        검증을 위한 참고문서 반환
    """

    vectorstore = load_vector_from_local()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs= {
            "k": 10
        }
    )
    filtered_docs = [doc for doc in retriever.invoke("id: 26, id: 30, id: 39") 
                     if doc.metadata.get('id') in [26, 30, 39]]
    
    context = format_docs(filtered_docs)

    return context



def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "관련 문서를 찾지 못했습니다"
    
    sections = []

    for doc in docs:
        id = doc.metadata.get('id')
        section = f"id = {id}"
        section += f"\n{doc.page_content}"
        sections.append(section)
    
    return "\n\n".join(sections)