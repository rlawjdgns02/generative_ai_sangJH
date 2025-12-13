# -*- coding: utf-8 -*-
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.graph.agent import MovieChatAgent


def test_basic_chat():
    print("\n" + "="*60)
    print("Test 1: Basic Chat")
    print("="*60)

    agent = MovieChatAgent(enable_memory=False)

    questions = [
        "Hello!",
        "Who are you?",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        response = agent.get_response(q, [])
        print(f"AI: {response}")


def test_movie_search():
    print("\n" + "="*60)
    print("Test 2: Movie Search (Tool)")
    print("="*60)

    agent = MovieChatAgent(enable_memory=False)

    questions = [
        "Tell me about Interstellar",
        "Find Inception",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        response = agent.get_response(q, [])
        print(f"AI: {response}")


def test_movie_recommendation():
    print("\n" + "="*60)
    print("Test 3: Movie Recommendation (Tool)")
    print("="*60)

    agent = MovieChatAgent(enable_memory=False)

    questions = [
        "Recommend me SF movies",
        "Suggest 3 action movies",
    ]

    for q in questions:
        print(f"\nUser: {q}")
        response = agent.get_response(q, [])
        print(f"AI: {response}")


def test_rag_search():
    print("\n" + "="*60)
    print("Test 4: RAG Search (Real PDF Documents)")
    print("="*60)

    agent = MovieChatAgent(enable_memory=False)

    question = input("\nEnter your question: ").strip()

    if question:
        print(f"\nUser: {question}")
        response = agent.get_response(question, [])
        print(f"AI: {response}")
    else:
        print("No question entered.")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set!")
        return

    print("\n" + "="*60)
    print("Movie Chat Agent - Test Program")
    print("="*60)
    print("\n1. Basic Chat Test")
    print("2. Movie Search Test (Mock Data)")
    print("3. Movie Recommendation Test (Mock Data)")
    print("4. RAG Search Test (Real PDFs)")
    print("5. Run All Tests")
    print("0. Exit")

    while True:
        try:
            choice = input("\nSelect (0-5): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                test_basic_chat()
            elif choice == "2":
                test_movie_search()
            elif choice == "3":
                test_movie_recommendation()
            elif choice == "4":
                test_rag_search()
            elif choice == "5":
                test_basic_chat()
                test_movie_search()
                test_movie_recommendation()
                test_rag_search()
                print("\nAll tests completed!")
            else:
                print("Invalid choice (0-5)")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
