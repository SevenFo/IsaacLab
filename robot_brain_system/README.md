### system 状态机流程图
```
IDLE
  │
  │ execute_task()
  ▼
THINKING ────────► ERROR
  │
  │ plan success
  ▼
EXECUTING ◄───────────────────────────────────────┐
  │                                               │
  ├── skill_running: 监控、执行                    │
  │                                               │
  ├── skill_ended: _manage_skill_lifecycle        │
  │     │                                         │
  │     ├── has_pending_skills: _start_next_skill │
  │     │     │                                   │
  │     │     └── success ────────────────────────┘
  │     │
  │     └── no_pending_skills: → IDLE
  │
  └── human_feedback:
        ├── store feedback
        └── terminate_skill(INTERRUPTED)
              │
              └── skill_ended: → replan with feedback
```
