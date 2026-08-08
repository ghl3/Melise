export type Role = "user" | "assistant";

export interface Message {
  role: Role;
  content: string;
}

export interface ModelInfo {
  name: string;
  stage: "rlvr" | "sft" | "pretrain";
  params: number;
  tokenizer: string;
  vocab_size: number;
  max_seq_len: number;
  default: boolean;
}

export interface GenParams {
  temperature: number;
  top_k: number;
  max_tokens: number;
}

export interface DoneStats {
  model: string;
  prompt_tokens: number;
  tokens: number;
  tok_per_s: number;
  stopped: boolean;
  truncated: boolean;
  dropped_turns: number;
}
